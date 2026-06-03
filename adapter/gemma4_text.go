// SPDX-License-Identifier: MIT OR Apache-2.0

// Package adapter holds the protocol-specific output handling for models whose
// chat format carries reasoning channels or tool markers inline in the token
// stream (Gemma 4, Harmony). The streaming sessions that decode tokens are
// compute-gated and land with the model backend; the leaf string helpers here
// are pure and portable, so they go first.
package adapter

import (
	"regexp"
	"strings"
	"unicode/utf8"
)

// leadingThoughtPattern matches one or more reasoning blocks anchored to the
// start of the text: either a rendered <think>...</think> span or the raw
// <|channel>...<channel|> protocol form, separated by optional whitespace. The
// (?s) flag makes "." span newlines, and the non-greedy ".*?" stops at the first
// close tag, matching the reference re.DOTALL pattern exactly. It has no
// lookbehind or backreferences, so RE2 handles it.
var leadingThoughtPattern = regexp.MustCompile(`(?s)\A\s*(?:(?:<think>.*?</think>|<\|channel>.*?<channel\|>)\s*)+`)

// StripLeadingThoughts removes leading <think>...</think> or raw
// <|channel>...<channel|> reasoning spans from replayed assistant content.
//
// Gemma 4's multi-turn rule keeps only the final visible answer in chat
// history. Clients such as Open WebUI replay the full assistant content,
// including the rendered thought block (or the raw protocol form when a client
// preserves it). Feeding prior thought blocks back primes the model to emit
// malformed channel markers on the next turn, which then leak into user-facing
// output.
//
// The match is anchored to the start of the message, so an inline mention later
// in the text (an assistant explaining how <think> tags work) is left untouched.
// An unterminated block with no close tag does not match and passes through.
func StripLeadingThoughts(text string) string {
	if text == "" {
		return text
	}
	// The \A anchor means only the position at the start can match, so this
	// replaces at most once, mirroring the reference count=1.
	return leadingThoughtPattern.ReplaceAllString(text, "")
}

// MatchingPrefixLen returns the length of the longest suffix of text that is a
// prefix of marker, capped at len(marker)-1. A streaming parser uses it to hold
// back a partial marker at a buffer boundary: if the tail of the emitted text
// could be the start of a control marker, those bytes are kept buffered until
// the next chunk arrives rather than leaked as visible output. The full marker
// is never reported here (cap is len(marker)-1) because a complete marker is
// handled by the marker-matching path, not the partial-suffix path.
//
// marker is always an ASCII control token, so its prefix bytes line up with rune
// boundaries; text may carry multibyte runes, so the cap counts runes to mirror
// the reference len(), while the suffix test stays byte-exact.
func MatchingPrefixLen(text, marker string) int {
	maxLen := utf8.RuneCountInString(text)
	if n := len(marker) - 1; n < maxLen {
		maxLen = n
	}
	for size := maxLen; size > 0; size-- {
		if strings.HasSuffix(text, marker[:size]) {
			return size
		}
	}
	return 0
}
