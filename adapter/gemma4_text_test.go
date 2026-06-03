// SPDX-License-Identifier: MIT OR Apache-2.0

package adapter

import (
	"encoding/json"
	"os"
	"testing"
)

type gemma4TextFixture struct {
	Strip []struct {
		In  string `json:"in"`
		Out string `json:"out"`
	} `json:"strip"`
	Prefix []struct {
		Text   string `json:"text"`
		Marker string `json:"marker"`
		Out    int    `json:"out"`
	} `json:"prefix"`
}

func loadGemma4TextFixture(t *testing.T) gemma4TextFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/gemma4_text.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx gemma4TextFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return fx
}

func TestStripLeadingThoughts(t *testing.T) {
	fx := loadGemma4TextFixture(t)
	for i, c := range fx.Strip {
		if got := StripLeadingThoughts(c.In); got != c.Out {
			t.Errorf("strip[%d] in=%q:\n got %q\nwant %q", i, c.In, got, c.Out)
		}
	}
}

func TestMatchingPrefixLen(t *testing.T) {
	fx := loadGemma4TextFixture(t)
	for i, c := range fx.Prefix {
		if got := MatchingPrefixLen(c.Text, c.Marker); got != c.Out {
			t.Errorf("prefix[%d] text=%q marker=%q: got %d, want %d", i, c.Text, c.Marker, got, c.Out)
		}
	}
}

func TestMatchingPrefixLenMultibyte(t *testing.T) {
	// A multibyte rune at the tail cannot start an ASCII marker, so no partial
	// match is reported; the rune-counted cap stays faithful to the reference.
	if got := MatchingPrefixLen("café", "<channel|>"); got != 0 {
		t.Errorf("multibyte tail: got %d, want 0", got)
	}
	if got := MatchingPrefixLen("café<", "<channel|>"); got != 1 {
		t.Errorf("multibyte then marker start: got %d, want 1", got)
	}
}

func BenchmarkStripLeadingThoughts(b *testing.B) {
	in := "<think>\nstep one\nstep two reasoning across\nmultiple lines\n</think>\n\nThe final visible answer to the user."
	b.ReportAllocs()
	for b.Loop() {
		_ = StripLeadingThoughts(in)
	}
}

func BenchmarkMatchingPrefixLen(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = MatchingPrefixLen("some streamed text<channel", "<channel|>")
	}
}
