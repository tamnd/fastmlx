// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestInjectJSONInstructionParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "inject_json_instruction.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Messages    json.RawMessage `json:"messages"`
		Instruction string          `json:"instruction"`
		Result      json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		in, ok := parseOrdered(string(c.Messages))
		if !ok || in.kind != kindArray {
			t.Fatalf("case %d: messages not a JSON array: %s", i, c.Messages)
		}
		want, ok := parseOrdered(string(c.Result))
		if !ok {
			t.Fatalf("case %d: result not valid JSON: %s", i, c.Result)
		}
		got := jval{kind: kindArray, arr: InjectJSONInstruction(in.arr, c.Instruction)}
		if g, w := got.dump(), want.dump(); g != w {
			t.Errorf("case %d:\n got %s\nwant %s", i, g, w)
		}
	}
}

func TestShouldStoreResponseParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "should_store_response.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		StoreFlag json.RawMessage `json:"store_flag"`
		Result    bool            `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		flag, ok := parseOrdered(string(c.StoreFlag))
		if !ok {
			t.Fatalf("case %d: store_flag not valid JSON: %s", i, c.StoreFlag)
		}
		if got := ShouldStoreResponse(flag); got != c.Result {
			t.Errorf("case %d: ShouldStoreResponse(%s) = %v, want %v", i, c.StoreFlag, got, c.Result)
		}
	}
}

func BenchmarkInjectJSONInstruction(b *testing.B) {
	b.ReportAllocs()
	in, _ := parseOrdered(`[{"role":"system","content":"Be terse."},{"role":"user","content":"hi"}]`)
	msgs := in.arr
	for b.Loop() {
		_ = InjectJSONInstruction(msgs, "Output JSON only.")
	}
}
