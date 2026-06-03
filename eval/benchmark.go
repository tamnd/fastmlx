// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import "strings"

// This file defines the GPU-free benchmark abstraction and the pure cores of the
// individual accuracy benchmarks: turning a dataset item into a prompt, pulling
// the predicted answer out of a response, and scoring it. Loading the bundled
// data from disk, running the model, and aggregating per-run results are the
// caller's seams; an item is the parsed dataset record, a plain map so each
// benchmark can read the fields it needs.

// Message is one chat message handed to the engine.
type Message struct {
	Role    string
	Content string
}

// Item is a single parsed dataset record. The fields vary by benchmark, so it
// stays an untyped map the way the reference passes dicts around.
type Item = map[string]any

// Benchmark is the pure, GPU-free surface of an accuracy benchmark. The methods
// that touch the engine or the filesystem live on the caller's side.
type Benchmark interface {
	// Name is the benchmark's stable identifier.
	Name() string
	// QuickSize is the sample size for a quick run.
	QuickSize() int
	// MaxTokens is the generation budget per question.
	MaxTokens() int
	// FormatPrompt turns an item into the chat messages for the engine.
	FormatPrompt(item Item) []Message
	// ExtractAnswer pulls the predicted answer out of the response text.
	ExtractAnswer(response string, item Item) string
	// CheckAnswer reports whether the prediction matches the gold answer.
	CheckAnswer(predicted string, item Item) bool
	// Category is the per-item subject for category scoring, or "" when none.
	Category(item Item) string
}

// userMessage wraps a single prompt string as one user-role message, the shape
// every benchmark's FormatPrompt returns.
func userMessage(content string) []Message {
	return []Message{{Role: "user", Content: content}}
}

// itemStr reads a string field, or "" when absent or not a string.
func itemStr(item Item, key string) string {
	if v, ok := item[key].(string); ok {
		return v
	}
	return ""
}

// itemList reads a list field, or nil when absent or not a list.
func itemList(item Item, key string) []any {
	if v, ok := item[key].([]any); ok {
		return v
	}
	return nil
}

// itemInt reads an integer field, tolerating the float64 a JSON number decodes
// to, an int, or a numeric string. Anything else yields 0.
func itemInt(item Item, key string) int {
	switch v := item[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case string:
		return NormalizeNumberToInt(v)
	}
	return 0
}

// NormalizeNumberToInt parses a base-10 integer string, returning 0 on failure.
// It is the small helper the dataset loaders use for label fields stored as
// strings.
func NormalizeNumberToInt(s string) int {
	n := 0
	neg := false
	s = strings.TrimSpace(s)
	for i, c := range s {
		if i == 0 && (c == '-' || c == '+') {
			neg = c == '-'
			continue
		}
		if c < '0' || c > '9' {
			return 0
		}
		n = n*10 + int(c-'0')
	}
	if neg {
		return -n
	}
	return n
}

// letterFor returns the choice letter for a zero-based index, or "" when the
// index is past the lettered range, reproducing the reference's ANSWER_MAP.get.
func letterFor(index int) string {
	if index < 0 || index >= len(AnswerMap) {
		return ""
	}
	return AnswerMap[index]
}
