// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// runeTokenizer is the deterministic one-token-per-rune tokenizer used to drive
// the truncation policy in tests, matching the reference capture's CharTok. It
// carries the source runes so Decode returns the first-N-rune prefix the
// reference's tokenizer.decode(ids[:max]) produces.
type runeTokenizer struct{ runes []rune }

func (t runeTokenizer) Encode(text string) []int { return make([]int, len([]rune(text))) }

func (t runeTokenizer) Decode(tokens []int) string {
	n := len(tokens)
	if n > len(t.runes) {
		n = len(t.runes)
	}
	return string(t.runes[:n])
}

func TestTruncateToolResultParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "tool_result_truncate.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Text      string `json:"text"`
		MaxTokens int    `json:"max_tokens"`
		Result    string `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		tok := runeTokenizer{runes: []rune(c.Text)}
		got := TruncateToolResult(c.Text, c.MaxTokens, tok)
		if got != c.Result {
			t.Errorf("case %d: TruncateToolResult(%q, %d)\n got  %q\n want %q",
				i, c.Text, c.MaxTokens, got, c.Result)
		}
	}
}

func BenchmarkTruncateToolResult(b *testing.B) {
	b.ReportAllocs()
	text := "line one is here\nline two is also here and quite a bit longer than the budget"
	tok := runeTokenizer{runes: []rune(text)}
	for b.Loop() {
		_ = TruncateToolResult(text, 30, tok)
	}
}
