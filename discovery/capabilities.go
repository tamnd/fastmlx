// SPDX-License-Identifier: MIT OR Apache-2.0

package discovery

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// HasVisionSubconfig reports whether a parsed config.json carries a vision
// subconfig. vision_config / vit_config count by key presence alone (a null
// value still means the key is there), while mm_vision_tower must be truthy in
// the Python sense, so an empty string does not count.
func HasVisionSubconfig(config map[string]any) bool {
	if _, ok := config["vision_config"]; ok {
		return true
	}
	if _, ok := config["vit_config"]; ok {
		return true
	}
	return truthy(config["mm_vision_tower"])
}

// ArchitectureIndicatesCausalLM reports whether any architecture name contains
// "causallm" once lowercased, the marker for a decoder-only causal language
// model.
func ArchitectureIndicatesCausalLM(architectures []string) bool {
	for _, arch := range architectures {
		if strings.Contains(strings.ToLower(arch), "causallm") {
			return true
		}
	}
	return false
}

// ContextLengthFromConfigs resolves the declared context length from already
// parsed config.json and tokenizer_config.json maps (the file reads are the
// caller's seam; a nil map means the file was absent). Resolution order: the
// top-level context keys, then nested text_config / language_config, then
// tokenizer_config's model_max_length when it is a finite positive integer
// below the int(1e30) sentinel. Returns 0 when nothing usable is found. Maps
// must be decoded with decodeNumberMap so float values are rejected like
// Python's isinstance(int).
func ContextLengthFromConfigs(config, tokenizerConfig map[string]any) int {
	if config != nil {
		if v := pickContextKey(config); v > 0 {
			return v
		}
		for _, nest := range []string{"text_config", "language_config"} {
			if sub, ok := config[nest].(map[string]any); ok {
				if v := pickContextKey(sub); v > 0 {
					return v
				}
			}
		}
	}
	if tokenizerConfig != nil {
		if v, ok := asInt(tokenizerConfig["model_max_length"]); ok && v > 0 && v < tokenizerMaxLengthSentinel {
			return int(v)
		}
	}
	return 0
}

// DetectThinkingDefault reads a chat template's thinking default from its text.
// Returns true when thinking is on by default (the template only suppresses it
// on "enable_thinking is false", the Qwen pattern), false when it is off by
// default ("default(false)" or "enable_thinking)", the Gemma pattern), and nil
// when the template has no enable_thinking toggle at all.
func DetectThinkingDefault(templateText string) *bool {
	if templateText == "" || !strings.Contains(templateText, "enable_thinking") {
		return nil
	}
	if strings.Contains(templateText, "enable_thinking is false") {
		return boolPtr(true)
	}
	if strings.Contains(templateText, "default(false)") || strings.Contains(templateText, "enable_thinking)") {
		return boolPtr(false)
	}
	return nil
}

// DetectPreserveThinking reports whether a chat template references the
// preserve_thinking flag (Qwen 3.6+ keeps historical <think> blocks only when
// it is set, and stripping them breaks prefix-cache reuse). Returns true when
// present, nil otherwise.
func DetectPreserveThinking(templateText string) *bool {
	if templateText == "" || !strings.Contains(templateText, "preserve_thinking") {
		return nil
	}
	return boolPtr(true)
}

// ModelTemplateText resolves a model's chat template text from disk: the
// standalone chat_template.jinja first, then tokenizer_config.json's
// chat_template field. Returns "" when neither is available. This is the I/O
// seam for DetectThinkingDefault / DetectPreserveThinking.
func ModelTemplateText(modelPath string) string {
	if b, err := os.ReadFile(filepath.Join(modelPath, "chat_template.jinja")); err == nil {
		return string(b)
	}
	b, err := os.ReadFile(filepath.Join(modelPath, "tokenizer_config.json"))
	if err != nil {
		return ""
	}
	var tc struct {
		ChatTemplate string `json:"chat_template"`
	}
	if json.Unmarshal(b, &tc) != nil {
		return ""
	}
	return tc.ChatTemplate
}

// truthy mirrors Python's bool() for the JSON value types: nil, false, an empty
// string, a zero number, and an empty list/object are falsey; everything else
// is truthy.
func truthy(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case string:
		return x != ""
	case json.Number:
		if f, err := x.Float64(); err == nil {
			return f != 0
		}
		return x != ""
	case float64:
		return x != 0
	case []any:
		return len(x) > 0
	case map[string]any:
		return len(x) > 0
	default:
		return true
	}
}

func boolPtr(b bool) *bool { return &b }
