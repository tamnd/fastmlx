// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"regexp"
	"strings"
)

// ToolCallStreamFilter suppresses tool-call markup from streamed content deltas
// so a model's control envelopes never leak into assistant-visible text. It
// detects three start envelopes during streaming - tokenizer-defined delimiters,
// namespaced XML opens (<ns:tool_call>), and the high-confidence bracket forms
// ([Calling tool: name(...)] / [Tool call: name]) - and removes the markup,
// resuming normal streaming once a closed envelope ends.
//
// Suppression is envelope-bounded: prose before an open envelope keeps flowing,
// the envelope body is dropped, and visible text after the matching close
// continues. Partial markup at a chunk boundary is held back until the next
// chunk resolves it, and finish() drops any unresolved marker-like tail under
// strict clean-output mode.
//
// The tokenizer dependency is reduced to two injected marker strings: the open
// marker (tool_call_start) and its optional end marker (tool_call_end). An open
// marker with an end marker forms a bounded pair tried ahead of the built-in
// <tool_call> pair; an open marker without an end marker is a one-sided marker
// (for example Mistral "[TOOL_CALLS]") that suppresses everything after it.
type ToolCallStreamFilter struct {
	markerPairs          [][2]string
	suppressAfterMarkers []string

	buffer           string
	suppressingUntil string
	hasSuppressUntil bool
	suppressing      bool
}

// suppressPermanently is the sentinel close marker used by one-sided markers to
// request that all remaining content be dropped.
const suppressPermanently = "__suppress_permanently__"

var (
	streamNamespacedOpenRe = regexp.MustCompile(`<([A-Za-z_][\w.\-]*):tool_call>`)
	streamBracketCallRe    = regexp.MustCompile(`(?s)^\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.\-]*)(?:\((\{.*?\})\))?\]`)
	streamNameRe           = regexp.MustCompile(`^[A-Za-z_][\w.\-]*$`)
)

// NewToolCallStreamFilter builds a filter from the tokenizer's open/end markers.
// Empty strings mean "no tokenizer marker"; an empty markerStart leaves only the
// built-in <tool_call> pair, and a non-empty markerStart with an empty markerEnd
// registers a one-sided suppress-after marker.
func NewToolCallStreamFilter(markerStart, markerEnd string) *ToolCallStreamFilter {
	f := &ToolCallStreamFilter{
		markerPairs: [][2]string{{"<tool_call>", "</tool_call>"}},
	}
	if markerStart != "" {
		if markerEnd != "" {
			f.markerPairs = append([][2]string{{markerStart, markerEnd}}, f.markerPairs...)
		} else {
			f.suppressAfterMarkers = append(f.suppressAfterMarkers, markerStart)
		}
	}
	return f
}

// startEnvelope is the result of locating the earliest opening envelope. hasClose
// false means the whole envelope is already consumed (no close to wait for);
// otherwise closeMarker is the marker to suppress up to, which may be the
// suppressPermanently sentinel.
type startEnvelope struct {
	idx         int
	consumeLen  int
	closeMarker string
	hasClose    bool
}

// findStartEnvelope returns the earliest complete opening envelope in text, or
// nil when none is present. On ties it returns the first one found in the order
// markers, namespaced open, bracket forms, suppress-after markers.
func (f *ToolCallStreamFilter) findStartEnvelope(text string) *startEnvelope {
	var starts []startEnvelope

	for _, pair := range f.markerPairs {
		if idx := strings.Index(text, pair[0]); idx >= 0 {
			starts = append(starts, startEnvelope{idx: idx, consumeLen: len(pair[0]), closeMarker: pair[1], hasClose: true})
		}
	}

	if m := streamNamespacedOpenRe.FindStringSubmatchIndex(text); m != nil {
		ns := text[m[2]:m[3]]
		starts = append(starts, startEnvelope{idx: m[0], consumeLen: m[1] - m[0], closeMarker: "</" + ns + ":tool_call>", hasClose: true})
	}

	for _, bp := range f.bracketPrefixes() {
		bracketIdx := strings.Index(text, bp)
		for bracketIdx >= 0 {
			candidate := text[bracketIdx:]
			if loc := streamBracketCallRe.FindStringIndex(candidate); loc != nil {
				starts = append(starts, startEnvelope{idx: bracketIdx, consumeLen: loc[1], hasClose: false})
			}
			next := strings.Index(text[bracketIdx+1:], bp)
			if next < 0 {
				break
			}
			bracketIdx = bracketIdx + 1 + next
		}
	}

	for _, sa := range f.suppressAfterMarkers {
		if idx := strings.Index(text, sa); idx >= 0 {
			starts = append(starts, startEnvelope{idx: idx, consumeLen: len(text) - idx, closeMarker: suppressPermanently, hasClose: true})
		}
	}

	if len(starts) == 0 {
		return nil
	}
	best := starts[0]
	for _, s := range starts[1:] {
		if s.idx < best.idx {
			best = s
		}
	}
	return &best
}

// bracketPrefixes is the fixed set of high-confidence bracket tool-call prefixes.
func (f *ToolCallStreamFilter) bracketPrefixes() []string {
	return []string{"[Calling tool:", "[Tool call:"}
}

// partialPrefixLen returns the length of the longest suffix of text that is a
// proper prefix of marker.
func partialPrefixLen(text, marker string) int {
	maxLen := len(text)
	if len(marker)-1 < maxLen {
		maxLen = len(marker) - 1
	}
	for n := maxLen; n > 0; n-- {
		if strings.HasSuffix(text, marker[:n]) {
			return n
		}
	}
	return 0
}

// couldBePartialNamespacedOpen reports whether candidate could be the prefix of a
// namespaced <ns:tool_call> open tag.
func couldBePartialNamespacedOpen(candidate string) bool {
	if !strings.HasPrefix(candidate, "<") {
		return false
	}
	if strings.Contains(candidate, ">") {
		return false
	}
	body := candidate[1:]
	if body == "" {
		return true
	}
	if strings.HasPrefix(body, "/") {
		return false
	}
	if !strings.Contains(body, ":") {
		return streamNameRe.MatchString(body)
	}
	parts := strings.SplitN(body, ":", 2)
	ns, suffix := parts[0], parts[1]
	if !streamNameRe.MatchString(ns) {
		return false
	}
	return strings.HasPrefix("tool_call", suffix)
}

// partialSuffixLen returns the length of the trailing suffix that might be the
// prefix of an opening marker and so must be held back until more text arrives.
func (f *ToolCallStreamFilter) partialSuffixLen(text string) int {
	keep := 0
	for _, pair := range f.markerPairs {
		if n := partialPrefixLen(text, pair[0]); n > keep {
			keep = n
		}
	}

	if lastLt := strings.LastIndex(text, "<"); lastLt >= 0 {
		candidate := text[lastLt:]
		if couldBePartialNamespacedOpen(candidate) && len(candidate) > keep {
			keep = len(candidate)
		}
	}

	for _, bp := range f.bracketPrefixes() {
		if n := partialPrefixLen(text, bp); n > keep {
			keep = n
		}
	}
	for _, sa := range f.suppressAfterMarkers {
		if n := partialPrefixLen(text, sa); n > keep {
			keep = n
		}
	}

	bracketIdx := -1
	for _, bp := range f.bracketPrefixes() {
		if idx := strings.LastIndex(text, bp); idx > bracketIdx {
			bracketIdx = idx
		}
	}
	if bracketIdx >= 0 {
		bracketCandidate := text[bracketIdx:]
		if !strings.Contains(bracketCandidate, "]") {
			// Hold the unresolved bracket prefix until it can be classified as a
			// parseable envelope or literal prose; never cap it, since capping
			// could leak raw control markup once the prefix grows past the cap.
			if len(bracketCandidate) > keep {
				keep = len(bracketCandidate)
			}
			return keep
		}
	}

	// Cap the retained suffix window to avoid unbounded buffering on malformed
	// text.
	if keep > 128 {
		return 128
	}
	return keep
}

// shouldDropTailAtFinish reports whether an unresolved tail should be suppressed
// under strict clean-output mode rather than emitted.
func (f *ToolCallStreamFilter) shouldDropTailAtFinish(tail string) bool {
	if tail == "" {
		return false
	}
	for _, pair := range f.markerPairs {
		if strings.HasPrefix(pair[0], tail) {
			return true
		}
	}
	for _, bp := range f.bracketPrefixes() {
		if strings.HasPrefix(tail, bp) {
			return true
		}
	}
	for _, sa := range f.suppressAfterMarkers {
		if strings.HasPrefix(sa, tail) || strings.HasPrefix(tail, sa) {
			return true
		}
	}
	if !strings.HasPrefix(tail, "<") {
		return false
	}
	if strings.Contains(tail, ">") {
		return false
	}
	body := tail[1:]
	if body == "" {
		return true
	}
	if strings.HasPrefix(body, "/") {
		return false
	}
	if !strings.Contains(body, ":") {
		// Preserve plain literal tails like "<alpha".
		return false
	}
	parts := strings.SplitN(body, ":", 2)
	ns, suffix := parts[0], parts[1]
	if !streamNameRe.MatchString(ns) {
		return false
	}
	return strings.HasPrefix("tool_call", suffix)
}

// sanitizePrefixBeforeSuppression strips unresolved bracket-control prefixes from
// prose emitted ahead of a suppressed envelope while preserving balanced literal
// bracket text.
func (f *ToolCallStreamFilter) sanitizePrefixBeforeSuppression(text string) string {
	hasBracket := false
	for _, bp := range f.bracketPrefixes() {
		if strings.Contains(text, bp) {
			hasBracket = true
			break
		}
	}
	if !hasBracket {
		return text
	}

	var out strings.Builder
	cursor := 0
	for cursor < len(text) {
		bracketIdx := -1
		bracketPrefix := ""
		for _, bp := range f.bracketPrefixes() {
			rel := strings.Index(text[cursor:], bp)
			if rel < 0 {
				continue
			}
			idx := cursor + rel
			if bracketIdx < 0 || idx < bracketIdx {
				bracketIdx = idx
				bracketPrefix = bp
			}
		}
		if bracketIdx < 0 {
			out.WriteString(text[cursor:])
			break
		}

		out.WriteString(text[cursor:bracketIdx])
		afterPrefix := bracketIdx + len(bracketPrefix)
		rel := strings.Index(text[afterPrefix:], "]")
		if rel < 0 {
			// Drop only the marker token; keep following prose.
			cursor = afterPrefix
			continue
		}
		closeIdx := afterPrefix + rel
		// Preserve balanced literal bracket text that is not being suppressed.
		out.WriteString(text[bracketIdx : closeIdx+1])
		cursor = closeIdx + 1
	}
	return out.String()
}

// Feed consumes a content delta and returns the portion safe to emit now,
// holding back any markup or partial markup.
func (f *ToolCallStreamFilter) Feed(text string) string {
	if f.suppressing || text == "" {
		return ""
	}

	f.buffer += text
	var out strings.Builder

	for f.buffer != "" {
		if f.hasSuppressUntil && f.suppressingUntil == suppressPermanently {
			f.suppressing = true
			f.hasSuppressUntil = false
			f.suppressingUntil = ""
			f.buffer = ""
			break
		}

		if f.hasSuppressUntil {
			endIdx := strings.Index(f.buffer, f.suppressingUntil)
			if endIdx < 0 {
				keep := partialPrefixLen(f.buffer, f.suppressingUntil)
				if keep > 0 {
					f.buffer = f.buffer[len(f.buffer)-keep:]
				} else {
					f.buffer = ""
				}
				break
			}
			f.buffer = f.buffer[endIdx+len(f.suppressingUntil):]
			f.hasSuppressUntil = false
			f.suppressingUntil = ""
			continue
		}

		start := f.findStartEnvelope(f.buffer)
		if start != nil {
			if start.idx > 0 {
				out.WriteString(f.sanitizePrefixBeforeSuppression(f.buffer[:start.idx]))
			}
			f.buffer = f.buffer[start.idx+start.consumeLen:]
			if start.hasClose {
				f.suppressingUntil = start.closeMarker
				f.hasSuppressUntil = true
			}
			continue
		}

		keep := f.partialSuffixLen(f.buffer)
		if keep == 0 {
			out.WriteString(f.buffer)
			f.buffer = ""
			break
		}
		if len(f.buffer) > keep {
			out.WriteString(f.buffer[:len(f.buffer)-keep])
			f.buffer = f.buffer[len(f.buffer)-keep:]
		}
		break
	}

	return out.String()
}

// Finish flushes the remaining safe buffer at the end of the stream. Unresolved
// marker-like suffixes are dropped so partial control markup never leaks into
// user-visible text.
func (f *ToolCallStreamFilter) Finish() string {
	if f.suppressing || f.hasSuppressUntil {
		f.buffer = ""
		f.hasSuppressUntil = false
		f.suppressingUntil = ""
		return ""
	}

	keep := f.partialSuffixLen(f.buffer)
	if keep >= len(f.buffer) {
		tail := f.buffer
		f.buffer = ""
		if f.shouldDropTailAtFinish(tail) {
			return ""
		}
		return tail
	}

	var buf string
	if keep > 0 {
		buf = f.buffer[:len(f.buffer)-keep]
	} else {
		buf = f.buffer
	}
	f.buffer = ""
	return buf
}
