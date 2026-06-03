// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

type grammarResolveCase struct {
	Fn             string          `json:"fn"`
	StructuredOuts json.RawMessage `json:"structured_outputs"`
	GuidedGrammar  json.RawMessage `json:"guided_grammar"`
	ResponseFormat json.RawMessage `json:"response_format"`
	Out            json.RawMessage `json:"out"`
}

func loadGrammarResolveFixture(t *testing.T) []grammarResolveCase {
	t.Helper()
	data, err := os.ReadFile("testdata/grammar_resolve.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []grammarResolveCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	return cases
}

// rawJval parses a fixture field; a JSON null (or absent field) becomes jnull().
func rawJval(t *testing.T, raw json.RawMessage) jval {
	t.Helper()
	if len(raw) == 0 {
		return jnull()
	}
	v, ok := parseOrdered(string(raw))
	if !ok {
		t.Fatalf("bad fixture json: %s", raw)
	}
	return v
}

// rawString reads a fixture field that holds a JSON string or null.
func rawString(t *testing.T, raw json.RawMessage) string {
	t.Helper()
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("bad fixture string: %s", raw)
	}
	return s
}

func TestGrammarResolveParity(t *testing.T) {
	for i, c := range loadGrammarResolveFixture(t) {
		want := rawJval(t, c.Out)
		var got jval
		switch c.Fn {
		case "normalize":
			got = NormalizeStructuredOutputs(rawJval(t, c.StructuredOuts), rawString(t, c.GuidedGrammar))
		case "format":
			got = BuildFormatElement(rawJval(t, c.StructuredOuts), rawJval(t, c.ResponseFormat))
		default:
			t.Fatalf("case %d: unknown fn %q", i, c.Fn)
		}
		if got.dumpASCII() != want.dumpASCII() {
			t.Errorf("case %d (%s):\n got  %s\n want %s", i, c.Fn, got.dumpASCII(), want.dumpASCII())
		}
	}
}

func TestSettingsGuidedGrammar(t *testing.T) {
	cases := []struct {
		enabled bool
		grammar string
		want    string
	}{
		{false, "root ::= \"a\"", ""},
		{true, "", ""},
		{true, "   ", ""},
		{true, "  root ::= \"a\"  ", "root ::= \"a\""},
		{true, "root ::= \"a\"", "root ::= \"a\""},
	}
	for i, c := range cases {
		if got := SettingsGuidedGrammar(c.enabled, c.grammar); got != c.want {
			t.Errorf("case %d: got %q want %q", i, got, c.want)
		}
	}
}

func TestEffectiveGuidedGrammar(t *testing.T) {
	cases := []struct {
		soPresent, rfPresent bool
		req, settings, want  string
	}{
		{false, false, "req", "model", "req"},
		{true, true, "req", "model", "req"},
		{false, false, "", "model", "model"},
		{true, false, "", "model", ""},
		{false, true, "", "model", ""},
		{false, false, "", "", ""},
	}
	for i, c := range cases {
		if got := EffectiveGuidedGrammar(c.soPresent, c.rfPresent, c.req, c.settings); got != c.want {
			t.Errorf("case %d: got %q want %q", i, got, c.want)
		}
	}
}

func BenchmarkBuildFormatElement(b *testing.B) {
	so, _ := parseOrdered(`{"choice":["yes","no","maybe","café"]}`)
	b.ReportAllocs()
	for b.Loop() {
		_ = BuildFormatElement(so, jnull())
	}
}
