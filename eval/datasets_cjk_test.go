// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"encoding/json"
	"os"
	"testing"
)

type cjkFewShotFixture struct {
	DevItems  []map[string]any            `json:"dev_items"`
	FewShot   map[string][]map[string]any `json:"fewshot"`
	Normalize []normalizeCase             `json:"normalize"`
	Prompt    []promptCase                `json:"prompt"`
	Extract   []extractCase               `json:"extract"`
	Check     []checkCase                 `json:"check"`
	Category  []categoryCase              `json:"category"`
}

type cjkFixture struct {
	CMMLU cjkFewShotFixture `json:"cmmlu"`
	JMMLU benchFixture      `json:"jmmlu"`
	KMMLU cjkFewShotFixture `json:"kmmlu"`
}

func loadCJK(t *testing.T) cjkFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/datasets_cjk.json")
	if err != nil {
		t.Fatal(err)
	}
	var f cjkFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func devItemsOf(f cjkFewShotFixture) []Item {
	items := make([]Item, len(f.DevItems))
	copy(items, f.DevItems)
	return items
}

func TestCMMLUBuildFewShotParity(t *testing.T) {
	f := loadCJK(t).CMMLU
	got := BuildCMMLUFewShot(devItemsOf(f))
	if !jsonEqual(t, got, f.FewShot) {
		gb, _ := json.Marshal(got)
		wb, _ := json.Marshal(f.FewShot)
		t.Errorf("BuildCMMLUFewShot =\n%s\nwant\n%s", gb, wb)
	}
}

func TestCMMLUParity(t *testing.T) {
	f := loadCJK(t).CMMLU
	c := CMMLU{FewShot: BuildCMMLUFewShot(devItemsOf(f))}
	checkBenchmark(t, c, benchFixture{
		Prompt:   f.Prompt,
		Extract:  f.Extract,
		Check:    f.Check,
		Category: f.Category,
	})
}

func TestJMMLUParity(t *testing.T) {
	checkBenchmark(t, JMMLU{}, loadCJK(t).JMMLU)
}

func TestKMMLUNormalizeParity(t *testing.T) {
	f := loadCJK(t).KMMLU
	for i, c := range f.Normalize {
		got := NormalizeKMMLUItem(c.Raw)
		if !jsonEqual(t, got, c.Out) {
			gb, _ := json.Marshal(got)
			wb, _ := json.Marshal(c.Out)
			t.Errorf("NormalizeKMMLUItem case %d = %s, want %s", i, gb, wb)
		}
	}
}

func TestKMMLUBuildFewShotParity(t *testing.T) {
	f := loadCJK(t).KMMLU
	got := BuildKMMLUFewShot(devItemsOf(f))
	if !jsonEqual(t, got, f.FewShot) {
		gb, _ := json.Marshal(got)
		wb, _ := json.Marshal(f.FewShot)
		t.Errorf("BuildKMMLUFewShot =\n%s\nwant\n%s", gb, wb)
	}
}

func TestKMMLUParity(t *testing.T) {
	f := loadCJK(t).KMMLU
	k := KMMLU{FewShot: BuildKMMLUFewShot(devItemsOf(f))}
	checkBenchmark(t, k, benchFixture{
		Prompt:   f.Prompt,
		Extract:  f.Extract,
		Check:    f.Check,
		Category: f.Category,
	})
}

func BenchmarkCMMLUFormatPrompt(b *testing.B) {
	c := CMMLU{FewShot: map[string][]Item{
		"chinese_history": {
			{"question": "Dev q?", "choices": []any{"a", "b", "c", "d"}, "answer": "A"},
		},
	}}
	item := Item{
		"question": "天问题?",
		"choices":  []any{"春", "夏", "秋", "冬"},
		"answer":   "D",
		"subject":  "chinese_history",
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = c.FormatPrompt(item)
	}
}

func BenchmarkJMMLUFormatPrompt(b *testing.B) {
	item := Item{
		"question": "問一?",
		"choices":  []any{"春", "夏", "秋", "冬"},
		"answer":   "C",
		"subject":  "japanese_history",
	}
	bench := JMMLU{}
	b.ReportAllocs()
	for b.Loop() {
		_ = bench.FormatPrompt(item)
	}
}

func BenchmarkKMMLUFormatPrompt(b *testing.B) {
	k := KMMLU{FewShot: map[string][]Item{
		"korean-history": {
			{"question": "Dev q?", "choices": []any{"가", "나", "다", "라"}, "answer": "B"},
		},
	}}
	item := Item{
		"question": "본 문제?",
		"choices":  []any{"하나", "둘", "셋", "넷"},
		"answer":   "C",
		"subject":  "korean-history",
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = k.FormatPrompt(item)
	}
}
