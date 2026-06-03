// SPDX-License-Identifier: MIT OR Apache-2.0

package discovery

// This file ports the tokenizer-config builders that centralize the
// model-specific tokenizer fixes. They build on the model-family detection in
// formatdetect.go, so loading the tokenizer can apply the same overrides
// consistently across callers. Loading the tokenizer itself is the caller's seam.

const qwen3EOSToken = "<|im_end|>"

// GetTokenizerConfig builds the tokenizer configuration options for a model,
// starting from the trust_remote_code flag and applying the Qwen3 fix: Qwen3's
// eos_token regressed to <|endoftext|> while its chat template still emits
// <|im_end|>, so a Qwen3 model pins eos_token back to <|im_end|>.
func GetTokenizerConfig(modelName string, trustRemoteCode bool) map[string]any {
	config := map[string]any{"trust_remote_code": trustRemoteCode}
	if IsQwen3Model(modelName) {
		config["eos_token"] = qwen3EOSToken
	}
	return config
}

// ApplyQwen3Fix applies the Qwen3 eos_token fix to an existing tokenizer config,
// setting eos_token to <|im_end|> for a Qwen3 model and leaving the config
// untouched otherwise. The config is mutated in place and returned.
func ApplyQwen3Fix(tokenizerConfig map[string]any, modelName string) map[string]any {
	if IsQwen3Model(modelName) {
		tokenizerConfig["eos_token"] = qwen3EOSToken
	}
	return tokenizerConfig
}
