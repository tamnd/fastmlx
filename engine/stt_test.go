// SPDX-License-Identifier: MIT OR Apache-2.0

package engine

import (
	"encoding/json"
	"os"
	"testing"
)

type sttFixture struct {
	GenerateLanguage []struct {
		In  *string `json:"in"`
		Out *string `json:"out"`
	} `json:"generate_language"`
	LooksMissing []struct {
		In  string `json:"in"`
		Out bool   `json:"out"`
	} `json:"looks_missing"`
	Hint []struct {
		Model string `json:"model"`
		Out   string `json:"out"`
	} `json:"hint"`
	Wrap []struct {
		Model   string            `json:"model"`
		Message string            `json:"message"`
		Out     []json.RawMessage `json:"out"`
	} `json:"wrap"`
}

func loadSTTFixture(t *testing.T) sttFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/stt.json")
	if err != nil {
		t.Fatal(err)
	}
	var f sttFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestNormalizeSTTGenerateLanguageParity(t *testing.T) {
	for i, c := range loadSTTFixture(t).GenerateLanguage {
		// The reference returns None for a None or blank input; "" stands in.
		in := ""
		if c.In != nil {
			in = *c.In
		}
		want := ""
		if c.Out != nil {
			want = *c.Out
		}
		if got := NormalizeSTTGenerateLanguage(in); got != want {
			t.Errorf("case %d: NormalizeSTTGenerateLanguage(%q) = %q, want %q", i, in, got, want)
		}
	}
}

func TestLooksLikeMissingProcessorParity(t *testing.T) {
	for i, c := range loadSTTFixture(t).LooksMissing {
		if got := looksLikeMissingProcessor(c.In); got != c.Out {
			t.Errorf("case %d: looksLikeMissingProcessor(%q) = %v, want %v", i, c.In, got, c.Out)
		}
	}
}

func TestMissingProcessorHintParity(t *testing.T) {
	for i, c := range loadSTTFixture(t).Hint {
		if got := missingProcessorHint(c.Model); got != c.Out {
			t.Errorf("case %d: missingProcessorHint(%q) mismatch:\n got  %q\n want %q", i, c.Model, got, c.Out)
		}
	}
}

func TestWrapSTTLoadErrorParity(t *testing.T) {
	for i, c := range loadSTTFixture(t).Wrap {
		var wantMsg string
		var wantWrapped bool
		if err := json.Unmarshal(c.Out[0], &wantMsg); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(c.Out[1], &wantWrapped); err != nil {
			t.Fatal(err)
		}
		gotMsg, gotWrapped := WrapSTTLoadError(c.Model, c.Message)
		if gotMsg != wantMsg || gotWrapped != wantWrapped {
			t.Errorf("case %d: WrapSTTLoadError(%q, %q) = (%q, %v), want (%q, %v)",
				i, c.Model, c.Message, gotMsg, gotWrapped, wantMsg, wantWrapped)
		}
	}
}

func BenchmarkNormalizeSTTGenerateLanguage(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = NormalizeSTTGenerateLanguage(" JA ")
	}
}
