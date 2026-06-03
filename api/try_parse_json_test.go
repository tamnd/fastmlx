// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestTryParseJSONParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "try_parse_json.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Input       string          `json:"input"`
		ResultIsStr bool            `json:"result_is_str"`
		Result      json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		got := TryParseJSON(c.Input)

		if c.ResultIsStr {
			var want string
			if err := json.Unmarshal(c.Result, &want); err != nil {
				t.Fatalf("case %d: fixture result is not a string: %s", i, c.Result)
			}
			if got.kind != kindString || got.s != want {
				t.Errorf("case %d: TryParseJSON(%q) got kind=%c %q want string %q",
					i, c.Input, got.kind, got.s, want)
			}
			continue
		}

		if got.kind == kindString {
			t.Errorf("case %d: TryParseJSON(%q) got string %q want parsed %s",
				i, c.Input, got.s, c.Result)
			continue
		}
		want, ok := parseOrdered(string(c.Result))
		if !ok {
			t.Fatalf("case %d: fixture result is not valid JSON: %s", i, c.Result)
		}
		if g, w := got.dump(), want.dump(); g != w {
			t.Errorf("case %d: TryParseJSON(%q) got %q want %q", i, c.Input, g, w)
		}
	}
}

func BenchmarkTryParseJSON(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = TryParseJSON(`{"name": "search", "args": {"q": "x"}}`)
	}
}
