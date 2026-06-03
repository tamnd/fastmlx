// SPDX-License-Identifier: MIT OR Apache-2.0

// Package eval holds the GPU-free text parsers of the accuracy-evaluation
// harness: pulling a model's final answer out of free-form generation and
// scoring it against the gold label. Multiple-choice letter extraction, last
// code-block extraction, think-tag stripping, and the GSM8K numeric extraction
// and normalization all live here. Loading datasets, formatting few-shot
// prompts that depend on bundled data, and driving the model are not part of
// this package; the deterministic samplers are deferred because they reproduce
// Python's Mersenne Twister stream, which is a separate slice.
package eval

import (
	"encoding/json"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"unicode"
)

// AnswerMap maps a zero-based choice index to its letter, as the reference does
// for four-option multiple choice.
var AnswerMap = []string{"A", "B", "C", "D"}

// mcRegexes caches the two compiled patterns per set of valid letters so a
// repeated call with the same letters does not recompile.
var mcRegexes sync.Map // string -> *mcPattern

type mcPattern struct {
	answer *regexp.Regexp
	letter *regexp.Regexp
}

func mcPatternFor(letters string) *mcPattern {
	if v, ok := mcRegexes.Load(letters); ok {
		return v.(*mcPattern)
	}
	p := &mcPattern{
		answer: regexp.MustCompile(`(?:answer\s*(?:is|:)\s*)([` + letters + `])\b`),
		letter: regexp.MustCompile(`\b([` + letters + `])\b`),
	}
	mcRegexes.Store(letters, p)
	return p
}

// ExtractMCAnswer pulls a single multiple-choice letter out of a response. It
// first looks for an explicit "answer is X" / "answer: X" phrasing and takes the
// last such match, then falls back to the last standalone valid letter, then to
// the first character. The response is upper-cased before matching, so the
// lower-case "answer" cue in the first pattern never fires against the
// upper-cased text; this matches the reference behavior exactly, where the
// fallback to the last standalone letter does the real work.
func ExtractMCAnswer(response string, validLetters []string) string {
	upper := strings.ToUpper(strings.TrimSpace(response))
	letters := strings.Join(validLetters, "")
	p := mcPatternFor(letters)

	if m := p.answer.FindAllStringSubmatch(upper, -1); m != nil {
		return m[len(m)-1][1]
	}
	if m := p.letter.FindAllStringSubmatch(upper, -1); m != nil {
		return m[len(m)-1][1]
	}
	trimmed := strings.TrimSpace(response)
	if trimmed != "" {
		first := strings.ToUpper(trimmed[:1])
		if slices.Contains(validLetters, first) {
			return first
		}
	}
	return ""
}

var (
	pythonBlockRe  = regexp.MustCompile("(?s)```python\\s*\\n(.*?)```")
	genericBlockRe = regexp.MustCompile("(?s)```\\s*\\n(.*?)```")
)

// ExtractLastCodeBlock returns the last fenced code block in a response,
// preferring a ```python block, then any fenced block, taking the last match in
// each case so example or draft blocks earlier in the text do not win. When no
// fence is present it falls back to collecting lines from the first def/class/
// import/from/# line onward, and returns the whole response if even that finds
// nothing.
func ExtractLastCodeBlock(response string) string {
	response = strings.TrimSpace(response)

	if blocks := pythonBlockRe.FindAllStringSubmatch(response, -1); blocks != nil {
		return strings.TrimSpace(blocks[len(blocks)-1][1])
	}
	if blocks := genericBlockRe.FindAllStringSubmatch(response, -1); blocks != nil {
		return strings.TrimSpace(blocks[len(blocks)-1][1])
	}

	var codeLines []string
	inCode := false
	for line := range strings.SplitSeq(response, "\n") {
		if !inCode && (strings.HasPrefix(line, "def ") ||
			strings.HasPrefix(line, "class ") ||
			strings.HasPrefix(line, "import ") ||
			strings.HasPrefix(line, "from ") ||
			strings.HasPrefix(line, "#")) {
			inCode = true
		}
		if inCode {
			codeLines = append(codeLines, line)
		}
	}
	if len(codeLines) > 0 {
		return strings.Join(codeLines, "\n")
	}
	return response
}

var thinkTagRe = regexp.MustCompile(`(?s)<think>.*?</think>`)

// StripThinkTags removes every <think>...</think> block from model output and
// trims the surrounding whitespace.
func StripThinkTags(text string) string {
	return strings.TrimSpace(thinkTagRe.ReplaceAllString(text, ""))
}

var (
	hashAnswerRe = regexp.MustCompile(`####\s*(-?[\d,]+(?:\.\d+)?)`)
	numberRe     = regexp.MustCompile(`-?[\d,]+(?:\.\d+)?`)
)

// ExtractNumericAnswer pulls the final numeric answer out of a GSM8K-style
// response, preferring the value after a "####" marker and otherwise taking the
// last number in the text, with any thousands commas removed.
func ExtractNumericAnswer(text string) string {
	if m := hashAnswerRe.FindStringSubmatch(text); m != nil {
		return strings.ReplaceAll(m[1], ",", "")
	}
	if nums := numberRe.FindAllString(text, -1); nums != nil {
		return strings.ReplaceAll(nums[len(nums)-1], ",", "")
	}
	return ""
}

// NormalizeNumber canonicalizes a numeric string for comparison: it strips
// whitespace and commas, and if the value is integral renders it without a
// fractional part. A string that does not parse as a finite number is returned
// stripped but otherwise unchanged, matching the reference, which leaves
// unparseable or overflowing values alone.
func NormalizeNumber(s string) string {
	s = strings.ReplaceAll(strings.TrimSpace(s), ",", "")
	val, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsInf(val, 0) || math.IsNaN(val) {
		return s
	}
	if val == math.Trunc(val) {
		return strconv.FormatInt(int64(val), 10)
	}
	return strconv.FormatFloat(val, 'g', -1, 64)
}

// CheckNumericAnswer reports whether a predicted numeric answer matches the gold
// answer once both are normalized. An empty prediction never matches.
func CheckNumericAnswer(predicted, answer string) bool {
	if predicted == "" {
		return false
	}
	return NormalizeNumber(predicted) == NormalizeNumber(answer)
}

// FormatSubjectName turns a subject slug like "high_school_biology" into a
// readable title like "High School Biology".
func FormatSubjectName(subject string) string {
	return pyTitle(strings.ReplaceAll(subject, "_", " "))
}

// pyTitle reproduces Python's str.title(): the first letter of each run of
// letters is upper-cased and the rest lower-cased, with any non-letter acting as
// a word boundary.
func pyTitle(s string) string {
	out := []rune(s)
	prevLetter := false
	for i, r := range out {
		letter := unicode.IsLetter(r)
		if letter && !prevLetter {
			out[i] = unicode.ToUpper(r)
		} else if letter {
			out[i] = unicode.ToLower(r)
		}
		prevLetter = letter
	}
	return string(out)
}

// ParseChoices coerces a choices field, which may already be a list or a
// string repr of a list, into a slice. A genuine list is returned as-is; a
// string is parsed after swapping single quotes for double quotes; anything
// else, or a string that does not parse to a list, yields an empty slice.
func ParseChoices(field any) []any {
	switch v := field.(type) {
	case []any:
		return v
	case string:
		var parsed any
		if err := json.Unmarshal([]byte(strings.ReplaceAll(v, "'", `"`)), &parsed); err == nil {
			if list, ok := parsed.([]any); ok {
				return list
			}
		}
	}
	return []any{}
}

// FormatQuestion renders a multiple-choice question with its lettered options,
// one per line, as the reference prompt builder does.
func FormatQuestion(question string, choices []any) string {
	parts := []string{question}
	for i, choice := range choices {
		if i >= len(AnswerMap) {
			break
		}
		parts = append(parts, AnswerMap[i]+". "+toStr(choice))
	}
	return strings.Join(parts, "\n")
}

func toStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	b, _ := json.Marshal(v)
	return string(b)
}
