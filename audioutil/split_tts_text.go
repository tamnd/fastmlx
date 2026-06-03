// SPDX-License-Identifier: MIT OR Apache-2.0

package audioutil

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// sentenceEnd is the [.!?。！？] class the reference uses to find sentence
// boundaries, and clauseEnd is the [,;:，；：] class it uses to break an
// over-long sentence at clause boundaries. Each set carries the ASCII and the
// fullwidth CJK punctuation so mixed-language input splits the same way.
var sentenceEnd = map[rune]bool{'.': true, '!': true, '?': true, '。': true, '！': true, '？': true}

var clauseEnd = map[rune]bool{',': true, ';': true, ':': true, '，': true, '；': true, '：': true}

// SplitTTSText breaks text into conservative sentence-like chunks no longer than
// maxChars, porting _split_tts_text from audio_routes.py. It is the GPU-free
// chunker the TTS path runs before handing each chunk to the model, so a long
// document streams as a sequence of utterances.
//
// The reference splits on sentence punctuation, greedily packs whole sentences
// up to the budget, and only when a single sentence still overflows does it fall
// back to splitting on clause punctuation and, as a last resort, hard-cutting a
// token longer than the budget. Lengths are counted in code points and slices
// are taken on code points, matching Python str semantics; whitespace stripping
// uses Unicode whitespace.
//
// The reference's two splits use regex lookbehind, which RE2 does not support,
// so they are reimplemented directly: splitSentences cuts on a whitespace run
// that follows sentence punctuation or on a run of newlines, and splitClauses
// cuts immediately after each clause-punctuation character (the punctuation
// stays on the left piece) and drops the whitespace that follows it. The
// Unicode whitespace classes are RE2/unicode.IsSpace rather than Python's `\s`,
// equivalent for the ASCII and CJK text this handles.
func SplitTTSText(text string, maxChars int) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return []string{}
	}

	sentences := keepNonEmpty(splitSentences([]rune(text)))
	if len(sentences) == 0 {
		sentences = []string{text}
	}

	var chunks []string
	current := ""
	flush := func() {
		if current != "" {
			chunks = append(chunks, strings.TrimSpace(current))
			current = ""
		}
	}

	for _, sentence := range sentences {
		if utf8.RuneCountInString(sentence) > maxChars {
			flush()
			parts := keepNonEmpty(splitClauses([]rune(sentence)))
			if len(parts) == 0 {
				parts = []string{sentence}
			}
			buffer := ""
			for _, part := range parts {
				pr := []rune(part)
				for len(pr) > maxChars {
					if buffer != "" {
						chunks = append(chunks, strings.TrimSpace(buffer))
						buffer = ""
					}
					chunks = append(chunks, strings.TrimSpace(string(pr[:maxChars])))
					pr = []rune(strings.TrimSpace(string(pr[maxChars:])))
				}
				part = string(pr)
				if part == "" {
					continue
				}
				candidate := part
				if buffer != "" {
					candidate = strings.TrimSpace(buffer + " " + part)
				}
				if utf8.RuneCountInString(candidate) <= maxChars {
					buffer = candidate
				} else {
					if buffer != "" {
						chunks = append(chunks, strings.TrimSpace(buffer))
					}
					buffer = part
				}
			}
			if buffer != "" {
				chunks = append(chunks, strings.TrimSpace(buffer))
			}
			continue
		}

		candidate := sentence
		if current != "" {
			candidate = strings.TrimSpace(current + " " + sentence)
		}
		if current != "" && utf8.RuneCountInString(candidate) > maxChars {
			flush()
			current = sentence
		} else {
			current = candidate
		}
	}
	flush()

	if len(chunks) == 0 {
		return []string{text}
	}
	return chunks
}

// keepNonEmpty mirrors the reference comprehension [x.strip() for x in xs if x
// and x.strip()]: each piece is trimmed and dropped when empty.
func keepNonEmpty(pieces []string) []string {
	out := make([]string, 0, len(pieces))
	for _, p := range pieces {
		if p == "" {
			continue
		}
		if s := strings.TrimSpace(p); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// splitSentences reimplements re.split(r"(?<=[.!?。！？])\s+|\n+", text): it cuts
// on a whitespace run that immediately follows a sentence-ending character, or
// on a run of newlines anywhere. The first alternative wins when both apply, so
// a whitespace run after sentence punctuation is consumed whole even when it
// begins with newlines.
func splitSentences(rs []rune) []string {
	var segs []string
	n := len(rs)
	start, i := 0, 0
	for i < n {
		if i > 0 && sentenceEnd[rs[i-1]] && unicode.IsSpace(rs[i]) {
			j := i
			for j < n && unicode.IsSpace(rs[j]) {
				j++
			}
			segs = append(segs, string(rs[start:i]))
			start, i = j, j
			continue
		}
		if rs[i] == '\n' {
			j := i
			for j < n && rs[j] == '\n' {
				j++
			}
			segs = append(segs, string(rs[start:i]))
			start, i = j, j
			continue
		}
		i++
	}
	segs = append(segs, string(rs[start:]))
	return segs
}

// splitClauses reimplements re.split(r"(?<=[,;:，；：])\s*", sentence): it cuts
// immediately after every clause-punctuation character (which therefore stays on
// the left piece) and consumes the whitespace that follows the cut. The trailing
// piece after the final clause punctuation is emitted too, matching the empty
// match the zero-width regex produces at the end.
func splitClauses(rs []rune) []string {
	var segs []string
	n := len(rs)
	start, i := 0, 0
	for i < n {
		if clauseEnd[rs[i]] {
			end := i + 1
			segs = append(segs, string(rs[start:end]))
			j := end
			for j < n && unicode.IsSpace(rs[j]) {
				j++
			}
			start, i = j, j
			continue
		}
		i++
	}
	segs = append(segs, string(rs[start:]))
	return segs
}
