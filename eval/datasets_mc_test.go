// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"encoding/json"
	"os"
	"testing"
)

type fixtureMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type promptCase struct {
	Item map[string]any   `json:"item"`
	Out  []fixtureMessage `json:"out"`
}

type extractCase struct {
	Resp string         `json:"resp"`
	Item map[string]any `json:"item"`
	Out  string         `json:"out"`
}

type checkCase struct {
	Pred string         `json:"pred"`
	Item map[string]any `json:"item"`
	Out  bool           `json:"out"`
}

type categoryCase struct {
	Item map[string]any `json:"item"`
	Out  string         `json:"out"`
}

type benchFixture struct {
	Prompt   []promptCase   `json:"prompt"`
	Extract  []extractCase  `json:"extract"`
	Check    []checkCase    `json:"check"`
	Category []categoryCase `json:"category"`
}

type datasetsFixture struct {
	GSM8K     benchFixture `json:"gsm8k"`
	ARC       benchFixture `json:"arc"`
	HellaSwag benchFixture `json:"hellaswag"`
}

func loadDatasets(t *testing.T) datasetsFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/datasets_mc.json")
	if err != nil {
		t.Fatal(err)
	}
	var f datasetsFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func checkBenchmark(t *testing.T, b Benchmark, f benchFixture) {
	t.Helper()
	for i, c := range f.Prompt {
		got := b.FormatPrompt(c.Item)
		if len(got) != len(c.Out) {
			t.Errorf("%s FormatPrompt case %d: %d messages, want %d", b.Name(), i, len(got), len(c.Out))
			continue
		}
		for j, m := range got {
			if m.Role != c.Out[j].Role || m.Content != c.Out[j].Content {
				t.Errorf("%s FormatPrompt case %d msg %d =\n%q (%s)\nwant\n%q (%s)",
					b.Name(), i, j, m.Content, m.Role, c.Out[j].Content, c.Out[j].Role)
			}
		}
	}
	for i, c := range f.Extract {
		if got := b.ExtractAnswer(c.Resp, c.Item); got != c.Out {
			t.Errorf("%s ExtractAnswer case %d (%q) = %q, want %q", b.Name(), i, c.Resp, got, c.Out)
		}
	}
	for i, c := range f.Check {
		if got := b.CheckAnswer(c.Pred, c.Item); got != c.Out {
			t.Errorf("%s CheckAnswer case %d (pred %q) = %v, want %v", b.Name(), i, c.Pred, got, c.Out)
		}
	}
	for i, c := range f.Category {
		if got := b.Category(c.Item); got != c.Out {
			t.Errorf("%s Category case %d = %q, want %q", b.Name(), i, got, c.Out)
		}
	}
}

func TestGSM8KParity(t *testing.T)     { checkBenchmark(t, GSM8K{}, loadDatasets(t).GSM8K) }
func TestARCParity(t *testing.T)       { checkBenchmark(t, ARCChallenge{}, loadDatasets(t).ARC) }
func TestHellaSwagParity(t *testing.T) { checkBenchmark(t, HellaSwag{}, loadDatasets(t).HellaSwag) }

func BenchmarkGSM8KFormatPrompt(b *testing.B) {
	item := map[string]any{"question": "What is 2 plus 2?", "answer": "4"}
	bench := GSM8K{}
	b.ReportAllocs()
	for b.Loop() {
		_ = bench.FormatPrompt(item)
	}
}

func BenchmarkARCFormatPrompt(b *testing.B) {
	item := map[string]any{
		"question": "Which is a gas?",
		"choices":  []any{"rock", "water", "oxygen", "sand"},
		"labels":   []any{"A", "B", "C", "D"},
		"answer":   "C",
	}
	bench := ARCChallenge{}
	b.ReportAllocs()
	for b.Loop() {
		_ = bench.FormatPrompt(item)
	}
}
