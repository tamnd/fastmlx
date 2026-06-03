// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type shuffleCase struct {
	Seed uint64 `json:"seed"`
	N    int    `json:"n"`
	Out  []int  `json:"out"`
}

type tqNormalizeCase struct {
	Raw map[string]any `json:"raw"`
	Idx int            `json:"idx"`
	Ok  bool           `json:"ok"`
	Out map[string]any `json:"out"`
}

type truthfulQAFixture struct {
	Shuffle   []shuffleCase     `json:"shuffle"`
	Normalize []tqNormalizeCase `json:"normalize"`
}

func loadTruthfulQA(t *testing.T) truthfulQAFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/truthfulqa.json")
	if err != nil {
		t.Fatal(err)
	}
	var f truthfulQAFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestShuffleParity(t *testing.T) {
	for i, c := range loadTruthfulQA(t).Shuffle {
		idx := make([]int, c.N)
		for j := range idx {
			idx[j] = j
		}
		NewPyRandom(c.Seed).Shuffle(idx)
		want := c.Out
		if want == nil {
			want = []int{}
		}
		if !reflect.DeepEqual(idx, want) {
			t.Errorf("Shuffle case %d (seed %d, n %d) = %v, want %v", i, c.Seed, c.N, idx, want)
		}
	}
}

func TestNormalizeTruthfulQAItemParity(t *testing.T) {
	for i, c := range loadTruthfulQA(t).Normalize {
		got, ok := NormalizeTruthfulQAItem(c.Raw, c.Idx)
		if ok != c.Ok {
			t.Errorf("normalize case %d: ok = %v, want %v", i, ok, c.Ok)
			continue
		}
		if !ok {
			continue
		}
		if !reflect.DeepEqual(jsonRoundTrip(t, got), c.Out) {
			gb, _ := json.Marshal(got)
			wb, _ := json.Marshal(c.Out)
			t.Errorf("normalize case %d =\n%s\nwant\n%s", i, gb, wb)
		}
	}
}

func BenchmarkNormalizeTruthfulQAItem(b *testing.B) {
	raw := map[string]any{
		"question": "What is the capital of France?",
		"mc1_targets": map[string]any{
			"choices": []any{"Paris", "London", "Berlin", "Rome"},
			"labels":  []any{1.0, 0.0, 0.0, 0.0},
		},
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = NormalizeTruthfulQAItem(raw, 0)
	}
}
