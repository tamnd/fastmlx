// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "regexp"

// This file ports the tool-result wrapper the Harmony (gpt-oss) path uses when a
// result has been token-truncated. The Harmony chat template runs |tojson over
// tool-result content; when truncation breaks otherwise-valid JSON the content
// degrades to a bare string, and |tojson would then double-encode it (quote and
// escape the whole thing). Wrapping the truncated text in an object makes |tojson
// emit a clean JSON object instead. It is a pure transform; the token counting
// and truncation that produce the input stay on the tokenizer-gated path.

// truncationNoticePattern matches the truncation notice appended to a truncated
// tool result, anchored at the end of the text after a blank line.
var truncationNoticePattern = regexp.MustCompile(`\n\n<truncated total_tokens="(\d+)" shown_tokens="(\d+)" />\s*$`)

// WrapTruncatedForHarmony wraps a truncated tool result in an object so the
// Harmony template's |tojson filter produces a clean object rather than a
// double-encoded string. When the input ends with the truncation notice, the
// body before it lands under "output" and a human-readable "Showing X of Y
// tokens" summary lands under "truncated"; otherwise the whole text is the
// "output" with no summary.
func WrapTruncatedForHarmony(truncatedText string) jval {
	loc := truncationNoticePattern.FindStringSubmatchIndex(truncatedText)
	if loc == nil {
		return jobj("output", jstr(truncatedText))
	}
	body := truncatedText[:loc[0]]
	total := truncatedText[loc[2]:loc[3]]
	shown := truncatedText[loc[4]:loc[5]]
	summary := "Showing " + shown + " of " + total + " tokens"
	return jobj("output", jstr(body), "truncated", jstr(summary))
}
