// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

type openaiValidatorsFixture struct {
	Coerce []struct {
		Kind string          `json:"kind"`
		In   json.RawMessage `json:"in"`
		OK   bool            `json:"ok"`
		Out  *string         `json:"out"`
	} `json:"coerce"`
	Name []struct {
		Kind string          `json:"kind"`
		In   json.RawMessage `json:"in"`
		Out  json.RawMessage `json:"out"`
	} `json:"name"`
}

func loadOpenAIValidatorsFixture(t *testing.T) openaiValidatorsFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/openai_validators.json")
	if err != nil {
		t.Fatal(err)
	}
	var f openaiValidatorsFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// argFromFixture builds the argument jval, treating "str" cases as a raw string
// value (not parsed) so the string branch is exercised, and every other kind via
// parseOrdered.
func argFromFixture(t *testing.T, kind string, raw json.RawMessage) jval {
	t.Helper()
	if kind == "str" {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			t.Fatalf("bad string input %s: %v", raw, err)
		}
		return jval{kind: kindString, s: s}
	}
	v, ok := parseOrdered(string(raw))
	if !ok {
		t.Fatalf("bad input fixture %s", raw)
	}
	return v
}

func TestCoerceToolCallArgumentsParity(t *testing.T) {
	for i, c := range loadOpenAIValidatorsFixture(t).Coerce {
		arg := argFromFixture(t, c.Kind, c.In)
		got, err := CoerceToolCallArguments(arg)
		if c.OK {
			if err != nil {
				t.Errorf("case %d (%s): unexpected error %v for %s", i, c.Kind, err, c.In)
				continue
			}
			if c.Out == nil || got != *c.Out {
				t.Errorf("case %d (%s): CoerceToolCallArguments(%s) = %q, want %q", i, c.Kind, c.In, got, derefStr(c.Out))
			}
		} else if err == nil {
			t.Errorf("case %d (%s): CoerceToolCallArguments(%s) = %q, want error", i, c.Kind, c.In, got)
		}
	}
}

func TestNormalizeFunctionNameParity(t *testing.T) {
	for i, c := range loadOpenAIValidatorsFixture(t).Name {
		arg := argFromFixture(t, c.Kind, c.In)
		got := NormalizeFunctionName(arg)
		want, ok := parseOrdered(string(c.Out))
		if !ok {
			t.Fatalf("case %d: bad want fixture %s", i, c.Out)
		}
		if got.dumpASCII() != want.dumpASCII() {
			t.Errorf("case %d: NormalizeFunctionName(%s) = %s, want %s", i, c.In, got.dumpASCII(), want.dumpASCII())
		}
	}
}

func derefStr(s *string) string {
	if s == nil {
		return "<nil>"
	}
	return *s
}

func BenchmarkCoerceToolCallArguments(b *testing.B) {
	obj, _ := parseOrdered(`{"location":"Tokyo","count":3,"ratio":1.50}`)
	str := jval{kind: kindString, s: `{"location": "Tokyo", "count": 3}`}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = CoerceToolCallArguments(obj)
		_, _ = CoerceToolCallArguments(str)
	}
}

func BenchmarkNormalizeFunctionName(b *testing.B) {
	v := jval{kind: kindString, s: "  get_weather  "}
	b.ReportAllocs()
	for b.Loop() {
		_ = NormalizeFunctionName(v)
	}
}
