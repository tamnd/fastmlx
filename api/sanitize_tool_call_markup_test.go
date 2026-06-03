// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSanitizeToolCallMarkupParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "sanitize_tool_call_markup.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Text        string `json:"text"`
		Result      string `json:"result"`
		MarkerStart string `json:"marker_start"`
		MarkerEnd   string `json:"marker_end"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		got := SanitizeToolCallMarkup(c.Text, c.MarkerStart, c.MarkerEnd)
		if got != c.Result {
			t.Errorf("case %d: SanitizeToolCallMarkup(%q, %q, %q)\n got  %q\n want %q",
				i, c.Text, c.MarkerStart, c.MarkerEnd, got, c.Result)
		}
	}
}

func BenchmarkSanitizeToolCallMarkup(b *testing.B) {
	b.ReportAllocs()
	text := "before <tool_call>{\"name\":\"f\"}</tool_call> middle <weather:tool_call>{\"q\":\"x\"}</weather:tool_call> after"
	for b.Loop() {
		_ = SanitizeToolCallMarkup(text, "", "")
	}
}
