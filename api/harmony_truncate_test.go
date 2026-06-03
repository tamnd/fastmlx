// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

type wrapTruncatedFixture struct {
	Wrap []struct {
		In  string          `json:"in"`
		Out json.RawMessage `json:"out"`
	} `json:"wrap"`
}

func loadWrapTruncated(t *testing.T) wrapTruncatedFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/wrap_truncated.json")
	if err != nil {
		t.Fatal(err)
	}
	var f wrapTruncatedFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestWrapTruncatedForHarmonyParity(t *testing.T) {
	for i, c := range loadWrapTruncated(t).Wrap {
		got := WrapTruncatedForHarmony(c.In)
		want, ok := parseOrdered(string(c.Out))
		if !ok {
			t.Fatalf("case %d: bad output fixture", i)
		}
		if got.dumpASCII() != want.dumpASCII() {
			t.Errorf("case %d:\n got  %s\n want %s", i, got.dumpASCII(), want.dumpASCII())
		}
	}
}

func BenchmarkWrapTruncatedForHarmony(b *testing.B) {
	const in = "the visible tool output body\n\n<truncated total_tokens=\"5000\" shown_tokens=\"256\" />"
	b.ReportAllocs()
	for b.Loop() {
		_ = WrapTruncatedForHarmony(in)
	}
}
