// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"regexp"
	"strings"
)

// Reasoning models wrap their chain of thought in <think>...</think> tags.
// This file separates that reasoning from the visible answer, both for a
// complete response (ExtractThinking) and incrementally as tokens stream in
// (ThinkingParser). The streaming path is the historically bug-prone part:
// tags can split across chunks, and a model can open a think block and never
// close it, so the parser buffers partial tags and recovers a non-empty answer
// at the end.

const (
	openTag  = "<think>"
	closeTag = "</think>"
	openLen  = len(openTag)  // 7
	closeLen = len(closeTag) // 8
)

var (
	// Non-greedy, dot-matches-newline, leftmost: one think block at a time.
	thinkingPattern = regexp.MustCompile(`(?s)<think>(.*?)</think>`)
	// Implicit open: a close tag with no matching open earlier in the text.
	thinkingTailPattern = regexp.MustCompile(`(?s)^(.*?)</think>`)
)

// ExtractThinking splits complete model output into (reasoning, content).
//
// Cases handled:
//   - Normal: "<think>r</think>a" -> ("r", "a")
//   - No tags: "a" -> ("", "a"), returned verbatim without trimming
//   - Implicit open: "r</think>a" -> ("r", "a")
//   - Empty think: "<think></think>a" -> ("", "a")
//   - Think only: "<think>r</think>" -> ("r", "")
//   - Malformed open with no close: "<think>everything" -> ("", "everything"),
//     so a model that skips the closing token still yields a visible answer.
func ExtractThinking(text string) (thinking, content string) {
	if text == "" {
		return "", ""
	}

	var parts []string
	remaining := text
	for {
		loc := thinkingPattern.FindStringSubmatchIndex(remaining)
		if loc == nil {
			break
		}
		parts = append(parts, remaining[loc[2]:loc[3]])
		remaining = remaining[:loc[0]] + remaining[loc[1]:]
	}
	if len(parts) > 0 {
		return strings.TrimSpace(strings.Join(parts, "\n")), strings.TrimSpace(remaining)
	}

	// Content before a close tag that has no matching open tag.
	if strings.Contains(text, closeTag) && !strings.Contains(text, openTag) {
		loc := thinkingTailPattern.FindStringSubmatchIndex(text)
		if loc != nil {
			return strings.TrimSpace(text[loc[2]:loc[3]]), strings.TrimSpace(text[loc[1]:])
		}
	}

	// Open tag with no close: drop the open marker, treat the rest as content.
	if before, after, found := strings.Cut(text, openTag); found && !strings.Contains(text, closeTag) {
		return "", strings.TrimSpace(before + after)
	}

	// Tag-free text is content, returned verbatim.
	return "", text
}

// ThinkingParser is a stateful streaming parser. Each Feed returns the
// reasoning and content deltas extracted from that chunk; Finish flushes any
// buffered partial tag and recovers an answer when a think block never closed.
type ThinkingParser struct {
	inThinking          bool
	buffer              string
	closeSeen           bool
	thinkingAccumulated []string
	contentEmitted      bool
}

// NewThinkingParser builds a parser. startInThinking is set when the prompt
// already prepended the open tag, so the first tokens are reasoning.
func NewThinkingParser(startInThinking bool) *ThinkingParser {
	return &ThinkingParser{inThinking: startInThinking}
}

// Feed consumes a chunk and returns (thinkingDelta, contentDelta).
func (p *ThinkingParser) Feed(text string) (string, string) {
	if text == "" {
		return "", ""
	}

	text = p.buffer + text
	p.buffer = ""

	var thinkingOut, contentOut strings.Builder

	for i := 0; i < len(text); {
		if text[i] == '<' {
			remaining := text[i:]

			if strings.HasPrefix(remaining, openTag) {
				p.inThinking = true
				i += openLen
				continue
			}
			if strings.HasPrefix(remaining, closeTag) {
				p.inThinking = false
				p.closeSeen = true
				i += closeLen
				continue
			}
			if couldBeTag(remaining) {
				p.buffer = remaining
				break
			}
			if p.inThinking {
				thinkingOut.WriteByte('<')
			} else {
				contentOut.WriteByte('<')
			}
			i++
			continue
		}
		if p.inThinking {
			thinkingOut.WriteByte(text[i])
		} else {
			contentOut.WriteByte(text[i])
		}
		i++
	}

	thinkingDelta := thinkingOut.String()
	contentDelta := contentOut.String()
	if thinkingDelta != "" {
		p.thinkingAccumulated = append(p.thinkingAccumulated, thinkingDelta)
	}
	if contentDelta != "" {
		p.contentEmitted = true
	}
	return thinkingDelta, contentDelta
}

// Finish flushes the buffer at end of stream. When the model opened a think
// block, never closed it, and never produced content, the accumulated thinking
// text is re-emitted as content so the answer body is not empty.
func (p *ThinkingParser) Finish() (string, string) {
	partial := p.buffer
	p.buffer = ""

	if p.inThinking && !p.closeSeen && !p.contentEmitted && len(p.thinkingAccumulated) > 0 {
		recovered := strings.Join(p.thinkingAccumulated, "") + partial
		p.contentEmitted = true
		return "", recovered
	}

	if partial == "" {
		return "", ""
	}

	if p.inThinking {
		p.thinkingAccumulated = append(p.thinkingAccumulated, partial)
		return partial, ""
	}
	p.contentEmitted = true
	return "", partial
}

// couldBeTag reports whether s is a proper prefix of either tag but not yet a
// complete match, so the parser should buffer and wait for more input.
func couldBeTag(s string) bool {
	if len(s) >= closeLen {
		return false
	}
	if openTag[:len(s)] == s {
		return true
	}
	if closeTag[:len(s)] == s {
		return true
	}
	return false
}
