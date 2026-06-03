// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

func loadHarmonyPreprocess(t *testing.T) []struct {
	In  json.RawMessage `json:"in"`
	Out json.RawMessage `json:"out"`
} {
	t.Helper()
	data, err := os.ReadFile("testdata/harmony_preprocess.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		In  json.RawMessage `json:"in"`
		Out json.RawMessage `json:"out"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	return cases
}

func TestPreprocessHarmonyMessagesParity(t *testing.T) {
	for i, c := range loadHarmonyPreprocess(t) {
		in, ok := parseOrdered(string(c.In))
		if !ok {
			t.Fatalf("case %d: bad input fixture", i)
		}
		want, ok := parseOrdered(string(c.Out))
		if !ok {
			t.Fatalf("case %d: bad output fixture", i)
		}
		got := jval{kind: kindArray, arr: PreprocessHarmonyMessages(in.arr)}
		if got.dumpASCII() != want.dumpASCII() {
			t.Errorf("case %d:\n got  %s\n want %s", i, got.dumpASCII(), want.dumpASCII())
		}
	}
}

func TestPreprocessHarmonyMessagesEmpty(t *testing.T) {
	if got := PreprocessHarmonyMessages(nil); len(got) != 0 {
		t.Errorf("nil input: got %v, want empty", got)
	}
	if got := PreprocessHarmonyMessages([]jval{}); len(got) != 0 {
		t.Errorf("empty input: got %v, want empty", got)
	}
}

func TestPreprocessHarmonyMessagesNoMutateInput(t *testing.T) {
	in, _ := parseOrdered(`[{"role":"assistant","content":"<think>x</think>answer"}]`)
	before := in.dumpASCII()
	_ = PreprocessHarmonyMessages(in.arr)
	if after := in.dumpASCII(); after != before {
		t.Errorf("input mutated:\n before %s\n after  %s", before, after)
	}
}

func BenchmarkPreprocessHarmonyMessages(b *testing.B) {
	in, _ := parseOrdered(`[{"role":"system","content":"sys"},{"role":"user","content":"hi"},{"role":"assistant","content":"<think>a long chain of thought spanning\nseveral lines</think>The visible answer."},{"role":"tool","content":"tool out"}]`)
	b.ReportAllocs()
	for b.Loop() {
		_ = PreprocessHarmonyMessages(in.arr)
	}
}
