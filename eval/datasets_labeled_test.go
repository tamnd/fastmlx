// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"encoding/json"
	"os"
	"testing"
)

type labeledFixture struct {
	MMLUPro     benchFixture `json:"mmlu_pro"`
	MathQA      benchFixture `json:"mathqa"`
	SafetyBench benchFixture `json:"safetybench"`
	BBQ         benchFixture `json:"bbq"`
}

func loadLabeled(t *testing.T) labeledFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/datasets_labeled.json")
	if err != nil {
		t.Fatal(err)
	}
	var f labeledFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestMMLUProParity(t *testing.T) { checkBenchmark(t, MMLUPro{}, loadLabeled(t).MMLUPro) }
func TestMathQAParity(t *testing.T)  { checkBenchmark(t, MathQA{}, loadLabeled(t).MathQA) }
func TestSafetyBenchParity(t *testing.T) {
	checkBenchmark(t, SafetyBench{}, loadLabeled(t).SafetyBench)
}
func TestBBQParity(t *testing.T) { checkBenchmark(t, BBQ{}, loadLabeled(t).BBQ) }

func BenchmarkMMLUProFormatPrompt(b *testing.B) {
	item := map[string]any{
		"question": "Hard question?",
		"choices":  []any{"c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"},
		"labels":   []any{"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"},
		"answer":   "H",
		"subject":  "physics",
	}
	bench := MMLUPro{}
	b.ReportAllocs()
	for b.Loop() {
		_ = bench.FormatPrompt(item)
	}
}

func BenchmarkBBQFormatPrompt(b *testing.B) {
	item := map[string]any{
		"context":  "At the meeting.",
		"question": "Who led?",
		"choices":  []any{"X", "Y", "Unknown"},
		"labels":   []any{"A", "B", "C"},
		"answer":   "C",
		"category": "gender",
	}
	bench := BBQ{}
	b.ReportAllocs()
	for b.Loop() {
		_ = bench.FormatPrompt(item)
	}
}
