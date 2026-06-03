// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

func loadResponsesValidatorsFixture(t *testing.T) []struct {
	In  json.RawMessage `json:"in"`
	Out json.RawMessage `json:"out"`
} {
	t.Helper()
	data, err := os.ReadFile("testdata/responses_validators.json")
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

func TestSerializeComplexOutputParity(t *testing.T) {
	for i, c := range loadResponsesValidatorsFixture(t) {
		in, ok := parseOrdered(string(c.In))
		if !ok {
			t.Fatalf("case %d: bad input fixture %s", i, c.In)
		}
		want, ok := parseOrdered(string(c.Out))
		if !ok {
			t.Fatalf("case %d: bad output fixture %s", i, c.Out)
		}
		got := SerializeComplexOutput(in)
		if got.dumpASCII() != want.dumpASCII() {
			t.Errorf("case %d:\n got  %s\n want %s", i, got.dumpASCII(), want.dumpASCII())
		}
	}
}

func TestSerializeComplexOutputNoMutateInput(t *testing.T) {
	in, _ := parseOrdered(`{"type":"x","output":["a","b"]}`)
	before := in.dumpASCII()
	_ = SerializeComplexOutput(in)
	if after := in.dumpASCII(); after != before {
		t.Errorf("input mutated:\n before %s\n after  %s", before, after)
	}
}

func BenchmarkSerializeComplexOutput(b *testing.B) {
	item, _ := parseOrdered(`{"type":"function_call_output","call_id":"c1","output":[1,2,{"nested":[true,null]},"text"]}`)
	b.ReportAllocs()
	for b.Loop() {
		_ = SerializeComplexOutput(item)
	}
}
