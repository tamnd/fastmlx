// SPDX-License-Identifier: MIT OR Apache-2.0

package discovery

import (
	"encoding/json"
	"os"
	"testing"
)

func loadHFCacheFixture(t *testing.T) []struct {
	In  string  `json:"in"`
	Out *string `json:"out"`
} {
	t.Helper()
	data, err := os.ReadFile("testdata/hfcache.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		In  string  `json:"in"`
		Out *string `json:"out"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	return cases
}

func TestParseHFCacheModelNameParity(t *testing.T) {
	for i, c := range loadHFCacheFixture(t) {
		got, ok := ParseHFCacheModelName(c.In)
		if c.Out == nil {
			if ok {
				t.Errorf("case %d: ParseHFCacheModelName(%q) = (%q, true), want (_, false)", i, c.In, got)
			}
			continue
		}
		if !ok || got != *c.Out {
			t.Errorf("case %d: ParseHFCacheModelName(%q) = (%q, %v), want (%q, true)", i, c.In, got, ok, *c.Out)
		}
	}
}

func BenchmarkParseHFCacheModelName(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseHFCacheModelName("models--mlx-community--Llama-3.2-3B-Instruct-4bit")
	}
}
