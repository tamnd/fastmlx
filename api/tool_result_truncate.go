// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "fmt"

// ResultTokenizer is the tokenizer seam TruncateToolResult needs: the encode
// step counts tokens and the decode step turns a token-id prefix back into text.
// The real model tokenizer is injected here so the truncation policy stays
// testable without a GPU or model toolkit.
type ResultTokenizer interface {
	Encode(text string) []int
	Decode(tokens []int) string
}

// TruncateToolResult trims tool-result text to a token budget, ported from
// anthropic_utils.truncate_tool_result. Within budget the text is returned
// untouched. Over budget it decodes the first maxTokens tokens for an
// approximate cut, then prefers the last newline as a line boundary - but only
// when that boundary keeps more than half the decoded characters, so a newline
// near the start does not throw away most of the content. The kept text is
// re-encoded to report the true shown-token count, and a separate XML notice tag
// records the original and shown token counts.
//
// Character positions follow Python's code-point semantics (the newline search
// and the half-length guard count runes, and the cut lands on a rune boundary),
// so the result matches the reference even when the text carries multibyte runes.
func TruncateToolResult(text string, maxTokens int, tok ResultTokenizer) string {
	tokenIDs := tok.Encode(text)
	totalTokens := len(tokenIDs)

	if totalTokens <= maxTokens {
		return text
	}

	// Decode tokens up to the budget to get an approximate character position.
	truncated := tok.Decode(tokenIDs[:maxTokens])

	// Find the last newline for line-boundary truncation, taken only when it
	// does not discard more than half the decoded content.
	runes := []rune(truncated)
	lastNewline := lastRuneIndex(runes, '\n')
	if lastNewline > 0 && float64(lastNewline) > float64(len(runes))*0.5 {
		truncated = string(runes[:lastNewline])
	}

	// Recount actual tokens after the line-boundary adjustment.
	shownTokens := len(tok.Encode(truncated))

	notice := fmt.Sprintf("\n\n<truncated total_tokens=\"%d\" shown_tokens=\"%d\" />",
		totalTokens, shownTokens)

	return truncated + notice
}

// lastRuneIndex returns the index (in runes) of the last occurrence of r in
// runes, or -1 when absent.
func lastRuneIndex(runes []rune, r rune) int {
	for i := len(runes) - 1; i >= 0; i-- {
		if runes[i] == r {
			return i
		}
	}
	return -1
}
