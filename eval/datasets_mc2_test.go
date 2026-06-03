// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"encoding/json"
	"os"
	"testing"
)

type normalizeCase struct {
	Raw map[string]any `json:"raw"`
	Out map[string]any `json:"out"`
}

type mmluFixture struct {
	DevItems  []map[string]any            `json:"dev_items"`
	FewShot   map[string][]map[string]any `json:"fewshot"`
	Normalize []normalizeCase             `json:"normalize"`
	Prompt    []promptCase                `json:"prompt"`
	Extract   []extractCase               `json:"extract"`
	Check     []checkCase                 `json:"check"`
	Category  []categoryCase              `json:"category"`
}

type datasets2Fixture struct {
	MMLU       mmluFixture  `json:"mmlu"`
	Winogrande benchFixture `json:"winogrande"`
	TruthfulQA benchFixture `json:"truthfulqa"`
}

func loadDatasets2(t *testing.T) datasets2Fixture {
	t.Helper()
	data, err := os.ReadFile("testdata/datasets_mc2.json")
	if err != nil {
		t.Fatal(err)
	}
	var f datasets2Fixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// jsonEqual compares two values by their canonical JSON encoding, so map key
// order is insignificant.
func jsonEqual(t *testing.T, got, want any) bool {
	t.Helper()
	gb, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	wb, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	return string(gb) == string(wb)
}

func TestMMLUNormalizeParity(t *testing.T) {
	f := loadDatasets2(t)
	for i, c := range f.MMLU.Normalize {
		got := NormalizeMMLUItem(c.Raw)
		if !jsonEqual(t, got, c.Out) {
			gb, _ := json.Marshal(got)
			wb, _ := json.Marshal(c.Out)
			t.Errorf("NormalizeMMLUItem case %d = %s, want %s", i, gb, wb)
		}
	}
}

func TestMMLUBuildFewShotParity(t *testing.T) {
	f := loadDatasets2(t)
	devItems := make([]Item, len(f.MMLU.DevItems))
	copy(devItems, f.MMLU.DevItems)
	got := BuildMMLUFewShot(devItems)
	if !jsonEqual(t, got, f.MMLU.FewShot) {
		gb, _ := json.Marshal(got)
		wb, _ := json.Marshal(f.MMLU.FewShot)
		t.Errorf("BuildMMLUFewShot =\n%s\nwant\n%s", gb, wb)
	}
}

func TestMMLUParity(t *testing.T) {
	f := loadDatasets2(t)
	devItems := make([]Item, len(f.MMLU.DevItems))
	copy(devItems, f.MMLU.DevItems)
	m := MMLU{FewShot: BuildMMLUFewShot(devItems)}
	checkBenchmark(t, m, benchFixture{
		Prompt:   f.MMLU.Prompt,
		Extract:  f.MMLU.Extract,
		Check:    f.MMLU.Check,
		Category: f.MMLU.Category,
	})
}

func TestWinograndeParity(t *testing.T) {
	checkBenchmark(t, Winogrande{}, loadDatasets2(t).Winogrande)
}

func TestTruthfulQAParity(t *testing.T) {
	checkBenchmark(t, TruthfulQA{}, loadDatasets2(t).TruthfulQA)
}

func BenchmarkMMLUFormatPrompt(b *testing.B) {
	m := MMLU{FewShot: map[string][]Item{
		"high_school_biology": {
			{"question": "Dev q?", "choices": []any{"a", "b", "c", "d"}, "answer": "A"},
		},
	}}
	item := Item{
		"question": "What is a cell?",
		"choices":  []any{"A unit", "Rock", "Air", "Fire"},
		"answer":   "A",
		"subject":  "high_school_biology",
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = m.FormatPrompt(item)
	}
}
