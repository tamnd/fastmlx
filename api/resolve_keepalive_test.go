// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestResolveKeepaliveParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "resolve_keepalive.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Mode     string `json:"mode"`
		Protocol string `json:"protocol"`
		IsNone   bool   `json:"is_none"`
		Frame    string `json:"frame"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		frame, ok := ResolveKeepalive(c.Mode, c.Protocol)
		if ok == c.IsNone {
			t.Errorf("case %d: ResolveKeepalive(%q, %q) ok=%v want is_none=%v",
				i, c.Mode, c.Protocol, ok, c.IsNone)
		}
		if frame != c.Frame {
			t.Errorf("case %d: ResolveKeepalive(%q, %q) frame=%q want %q",
				i, c.Mode, c.Protocol, frame, c.Frame)
		}
	}
}

func BenchmarkResolveKeepalive(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_, _ = ResolveKeepalive("chunk", "openai_chat")
	}
}
