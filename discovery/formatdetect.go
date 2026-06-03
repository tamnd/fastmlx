// SPDX-License-Identifier: MIT OR Apache-2.0

package discovery

import "strings"

// This file ports the chat-format detectors that pick which parser and template
// fixups a model needs: Harmony (gpt-oss), Gemma 4, and Qwen3. Each combines the
// config.json model_type with a model-name fallback. The config read is the
// caller's seam: pass the raw model_type string (empty when there is no config
// or the field is absent), so the detection itself stays pure.

// IsHarmonyModel reports whether the model uses the Harmony format (gpt-oss). It
// is detected first by an exact model_type of "gpt_oss", then by a model name
// containing "gpt-oss" or "gptoss" (case-insensitive). Note the name fallback
// does not match the underscore form "gpt_oss".
func IsHarmonyModel(modelName, modelType string) bool {
	if modelType == "gpt_oss" {
		return true
	}
	if modelName != "" {
		nameLower := strings.ToLower(modelName)
		if strings.Contains(nameLower, "gpt-oss") || strings.Contains(nameLower, "gptoss") {
			return true
		}
	}
	return false
}

// IsGemma4Model reports whether the model is a Gemma 4 model: an exact
// model_type of "gemma4", or a name containing "gemma-4" or "gemma4"
// (case-insensitive).
func IsGemma4Model(modelName, modelType string) bool {
	if modelType == "gemma4" {
		return true
	}
	if modelName != "" {
		nameLower := strings.ToLower(modelName)
		if strings.Contains(nameLower, "gemma-4") || strings.Contains(nameLower, "gemma4") {
			return true
		}
	}
	return false
}

// IsQwen3Model reports whether the model is a Qwen3 model, by a name containing
// "qwen3" (case-insensitive).
func IsQwen3Model(modelName string) bool {
	return strings.Contains(strings.ToLower(modelName), "qwen3")
}
