// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestNormalizeResponseRecordParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "normalize_response_record.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		ResponseID   string          `json:"response_id"`
		ResponseData json.RawMessage `json:"response_data"`
		Result       json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		data, ok := parseOrdered(string(c.ResponseData))
		if !ok {
			t.Fatalf("case %d: response_data is not valid JSON: %s", i, c.ResponseData)
		}
		want, ok := parseOrdered(string(c.Result))
		if !ok {
			t.Fatalf("case %d: result is not valid JSON: %s", i, c.Result)
		}
		got := NormalizeResponseRecord(c.ResponseID, data)
		if g, w := got.dump(), want.dump(); g != w {
			t.Errorf("case %d (%s):\n got %s\nwant %s", i, c.ResponseID, g, w)
		}
	}
}

func BenchmarkNormalizeResponseRecord(b *testing.B) {
	b.ReportAllocs()
	data, _ := parseOrdered(`{"id":"resp_x","created_at":7,"output":[` +
		`{"type":"reasoning","summary":[{"text":"think"}]},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`)
	for b.Loop() {
		_ = NormalizeResponseRecord("resp_x", data)
	}
}
