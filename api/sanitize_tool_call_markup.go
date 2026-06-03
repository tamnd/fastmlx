// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "strings"

// SanitizeToolCallMarkup removes tool-call control markup from text while
// preserving the surrounding prose, ported from
// tool_calling.sanitize_tool_call_markup. It is the one-shot counterpart to the
// streaming ToolCallStreamFilter: an empty input yields an empty string, and any
// other input is run through the filter in full (Feed then Finish) and stripped
// of leading and trailing whitespace.
//
// The marker arguments carry the tokenizer-defined tool-call delimiters the
// reference reads from the tokenizer (tool_call_start and tool_call_end). They
// are injected here as plain strings so the policy stays GPU-free: an empty pair
// falls back to the built-in <tool_call>...</tool_call> envelope, a start with a
// matching end adds that pair, and a start with no end suppresses everything
// after the start marker.
func SanitizeToolCallMarkup(text, markerStart, markerEnd string) string {
	if text == "" {
		return ""
	}

	filter := NewToolCallStreamFilter(markerStart, markerEnd)
	cleaned := filter.Feed(text)
	cleaned += filter.Finish()
	return strings.TrimSpace(cleaned)
}
