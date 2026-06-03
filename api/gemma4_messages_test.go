// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestExtractGemma4MessagesParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "extract_gemma4_messages.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Messages json.RawMessage `json:"messages"`
		Result   json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		in, ok := parseOrdered(string(c.Messages))
		if !ok || in.kind != kindArray {
			t.Fatalf("case %d: messages is not a JSON array: %s", i, c.Messages)
		}
		want, ok := parseOrdered(string(c.Result))
		if !ok {
			t.Fatalf("case %d: result is not valid JSON: %s", i, c.Result)
		}
		got := jval{kind: kindArray, arr: ExtractGemma4Messages(in.arr)}
		if g, w := got.dump(), want.dump(); g != w {
			t.Errorf("case %d:\n got %s\nwant %s", i, g, w)
		}
	}
}

func BenchmarkExtractGemma4Messages(b *testing.B) {
	b.ReportAllocs()
	in, _ := parseOrdered(`[` +
		`{"role":"user","content":"weather?"},` +
		`{"role":"assistant","content":"<think>\nplan\n</think>\nok","tool_calls":[` +
		`{"id":"c1","function":{"name":"get_weather","arguments":"{\"city\": \"Seoul\"}"}}]},` +
		`{"role":"tool","tool_call_id":"c1","content":"{\"temp\": 21}"}]`)
	for b.Loop() {
		_ = ExtractGemma4Messages(in.arr)
	}
}
