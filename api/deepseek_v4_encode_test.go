// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestEncodeArgumentsToDSMLParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "encode_arguments_to_dsml.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Arguments string `json:"arguments"`
		Result    string `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		args, ok := parseOrdered(c.Arguments)
		if !ok {
			t.Fatalf("case %d: fixture arguments is not valid JSON: %s", i, c.Arguments)
		}
		if got := EncodeArgumentsToDSML(args); got != c.Result {
			t.Errorf("case %d: EncodeArgumentsToDSML(%q)\n got %q\nwant %q", i, c.Arguments, got, c.Result)
		}
	}
}

func BenchmarkEncodeArgumentsToDSML(b *testing.B) {
	b.ReportAllocs()
	args, _ := parseOrdered(`{"city": "Seoul", "unit": "celsius", "count": 3, "items": [1, 2, 3]}`)
	for b.Loop() {
		_ = EncodeArgumentsToDSML(args)
	}
}
