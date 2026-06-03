// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"maps"
	"strings"
)

// This file holds the pure request-shaping a run does around each engine call:
// resolving the token budget, joining the prompt text, building the forced
// sampling parameters, and deciding from a batch of raw outputs whether to
// auto-switch into thinking mode. The engine.chat call itself is the seam.

// Token budget for thinking and reasoning models (OpenCompass references 8K-32K).
const (
	ThinkingMinTokens = 8192
	ThinkingMaxTokens = 32768
)

// ResolveMaxTokens applies the benchmark's token-budget overrides. gpt_oss
// Harmony models get a quadrupled budget floored at 8192, since the analysis
// channel can consume the whole budget before the final channel is emitted;
// otherwise thinking mode clamps the budget into the thinking band. gpt_oss
// takes precedence over thinking mode.
func ResolveMaxTokens(base int, modelType string, enableThinking bool) int {
	if modelType == "gpt_oss" {
		return max(base*4, 8192)
	}
	if enableThinking {
		return min(max(base, ThinkingMinTokens), ThinkingMaxTokens)
	}
	return base
}

// PromptText joins the messages' contents with newlines, the text a run records
// as a question's prompt.
func PromptText(messages []Message) string {
	parts := make([]string, len(messages))
	for i, m := range messages {
		parts[i] = m.Content
	}
	return strings.Join(parts, "\n")
}

// HasThinkTags reports whether any raw model output carries a <think> tag, the
// signal a non-thinking run uses to auto-switch its first batch into thinking
// mode.
func HasThinkTags(rawTexts []string) bool {
	for _, raw := range rawTexts {
		if strings.Contains(raw, "<think>") {
			return true
		}
	}
	return false
}

// BuildSamplingKwargs assembles the engine request parameters for one question.
// It starts from the caller's sampling options, forces the resolved max-tokens
// budget and the deterministic decoding settings the benchmarks require, and
// folds enable_thinking into any existing chat_template_kwargs without dropping
// its other keys.
func BuildSamplingKwargs(sampling map[string]any, baseMaxTokens int, modelType string, enableThinking bool) map[string]any {
	kwargs := map[string]any{}
	maps.Copy(kwargs, sampling)

	kwargs["max_tokens"] = ResolveMaxTokens(baseMaxTokens, modelType, enableThinking)
	kwargs["temperature"] = 0.0
	kwargs["presence_penalty"] = 0.0
	kwargs["repetition_penalty"] = 1.0

	ct := map[string]any{}
	if existing, ok := kwargs["chat_template_kwargs"].(map[string]any); ok {
		maps.Copy(ct, existing)
	}
	ct["enable_thinking"] = enableThinking
	kwargs["chat_template_kwargs"] = ct

	return kwargs
}
