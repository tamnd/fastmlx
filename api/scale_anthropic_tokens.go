// SPDX-License-Identifier: MIT OR Apache-2.0

package api

// ScaleAnthropicTokens rescales a reported token count so Claude Code's
// auto-compact fires at the right moment when a model's real context window is
// smaller than the target the client assumes, porting scale_anthropic_tokens
// from server.py. The reference reads global settings and resolves the model's
// real window through get_max_context_window, both server-state-bound; those are
// injected here as scalingEnabled, targetContextSize, and actual (the resolved
// window, nil when no tier produced one), leaving only the pure formula.
//
// Scaling is skipped (the count passes through unchanged) when it is disabled
// (which also covers the reference's "no global settings" early return), when no
// real window resolved (nil or zero, matching `if not actual`), or when the real
// window already meets or exceeds the target. Otherwise the count is scaled by
// target/actual.
//
// The reference computes int(token_count * target_context_size / actual): the
// integer product is formed exactly first, then divided as a float, then
// truncated toward zero. This reproduces that order: the product is taken in
// int64, divided as float64, and converted back to int (Go's float-to-int
// conversion truncates toward zero like Python's int()).
func ScaleAnthropicTokens(tokenCount int, scalingEnabled bool, targetContextSize int, actual *int) int {
	if !scalingEnabled {
		return tokenCount
	}
	if actual == nil || *actual == 0 || *actual >= targetContextSize {
		return tokenCount
	}
	return int(float64(int64(tokenCount)*int64(targetContextSize)) / float64(*actual))
}
