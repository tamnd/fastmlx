// SPDX-License-Identifier: MIT OR Apache-2.0

package engine

import "strings"

// Pure helpers for the speech-to-text path: mapping request language codes to
// the names the audio backend expects, and turning an opaque model-load failure
// into an actionable message. The transcription forward pass itself runs on the
// audio engine behind the compute seam.

// isoToSTTLang maps OpenAI-style ISO language codes to the language names the
// mlx-audio STT backends accept.
var isoToSTTLang = map[string]string{
	"zh":  "chinese",
	"yue": "cantonese",
	"en":  "english",
	"de":  "german",
	"es":  "spanish",
	"fr":  "french",
	"it":  "italian",
	"pt":  "portuguese",
	"ru":  "russian",
	"ko":  "korean",
	"ja":  "japanese",
}

// NormalizeSTTGenerateLanguage maps an OpenAI-style ISO code to the language
// name the audio backend accepts. A code is matched case-insensitively after
// trimming; an empty or whitespace-only language yields "" (the reference's
// None, meaning no language is forced on generate), and an unrecognized code is
// returned trimmed but otherwise unchanged.
func NormalizeSTTGenerateLanguage(language string) string {
	normalized := strings.TrimSpace(language)
	if normalized == "" {
		return ""
	}
	if name, ok := isoToSTTLang[strings.ToLower(normalized)]; ok {
		return name
	}
	return normalized
}

// missingProcessorHints are the lowercase substrings whose presence in a load
// error points at a model that ships without its HuggingFace processor config.
var missingProcessorHints = []string{
	"preprocessor_config.json",
	"feature extractor",
	"featureextractor",
}

// looksLikeMissingProcessor reports whether a load-error message points at a
// missing processor / feature-extractor configuration.
func looksLikeMissingProcessor(message string) bool {
	lowered := strings.ToLower(message)
	for _, h := range missingProcessorHints {
		if strings.Contains(lowered, h) {
			return true
		}
	}
	return false
}

// missingProcessorHint is the actionable message explaining that an STT model
// is missing its HuggingFace processor / feature-extractor configuration.
func missingProcessorHint(modelName string) string {
	return "STT model '" + modelName + "' is missing the HuggingFace processor / " +
		"feature-extractor configuration (preprocessor_config.json and/or " +
		"tokenizer files). MLX-converted repositories sometimes omit these. " +
		"Fix: either use an HF-compatible variant of the model or copy " +
		"preprocessor_config.json, tokenizer.json and special_tokens_map.json " +
		"from the upstream HuggingFace repo into the local model directory."
}

// WrapSTTLoadError clarifies a known STT load failure. When the message points
// at a missing processor configuration it returns the actionable hint with the
// original error appended and true; otherwise it returns the message unchanged
// and false, so an unrecognized failure surfaces verbatim.
func WrapSTTLoadError(modelName, message string) (string, bool) {
	if looksLikeMissingProcessor(message) {
		return missingProcessorHint(modelName) + " Original error: " + message, true
	}
	return message, false
}
