// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type streamFilterCase struct {
	Marker    *string  `json:"marker"`
	MarkerEnd *string  `json:"marker_end"`
	Chunks    []string `json:"chunks"`
	Emitted   []string `json:"emitted"`
	Finish    string   `json:"finish"`
	Joined    string   `json:"joined"`
}

func loadStreamFilterCases(t *testing.T) []streamFilterCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "tool_call_stream_filter.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []streamFilterCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	return cases
}

func str(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

func TestToolCallStreamFilterParity(t *testing.T) {
	for i, c := range loadStreamFilterCases(t) {
		f := NewToolCallStreamFilter(str(c.Marker), str(c.MarkerEnd))
		emitted := make([]string, len(c.Chunks))
		for j, chunk := range c.Chunks {
			emitted[j] = f.Feed(chunk)
		}
		fin := f.Finish()

		if len(emitted) != len(c.Emitted) {
			t.Fatalf("case %d: emitted len %d, want %d", i, len(emitted), len(c.Emitted))
		}
		for j := range emitted {
			if emitted[j] != c.Emitted[j] {
				t.Errorf("case %d chunk %d: Feed = %q, want %q", i, j, emitted[j], c.Emitted[j])
			}
		}
		if fin != c.Finish {
			t.Errorf("case %d: Finish = %q, want %q", i, fin, c.Finish)
		}

		var joined string
		for _, e := range emitted {
			joined += e
		}
		joined += fin
		if joined != c.Joined {
			t.Errorf("case %d: joined = %q, want %q", i, joined, c.Joined)
		}
	}
}

func BenchmarkToolCallStreamFilterFeed(b *testing.B) {
	b.ReportAllocs()
	chunks := []string{"before <tool_", "call>{\"a\":1}</tool", "_call> after the call"}
	for b.Loop() {
		f := NewToolCallStreamFilter("", "")
		for _, c := range chunks {
			_ = f.Feed(c)
		}
		_ = f.Finish()
	}
}
