// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

// The fixtures in testdata/parity/toolcalls.json are captured from the Python
// reference generic tool-call parsers. Tool-call ids are random, so parity is
// checked on the cleaned text and on each call's (name, arguments) pair, where
// arguments must match the reference byte for byte including key order and the
// json.dumps space-after-separator style.

type toolCallExpect struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type toolCallFixture struct {
	Input     string           `json:"input"`
	Namespace string           `json:"namespace"`
	Cleaned   string           `json:"cleaned"`
	Calls     []toolCallExpect `json:"calls"`
}

type toolCallFixtures struct {
	XML        []toolCallFixture `json:"xml"`
	Namespaced []toolCallFixture `json:"namespaced"`
	Bracket    []toolCallFixture `json:"bracket"`
}

func loadToolCallFixtures(t testing.TB) toolCallFixtures {
	t.Helper()
	data, err := os.ReadFile("testdata/parity/toolcalls.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fx toolCallFixtures
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("decode fixtures: %v", err)
	}
	if len(fx.XML) == 0 || len(fx.Namespaced) == 0 || len(fx.Bracket) == 0 {
		t.Fatal("fixtures missing a format group")
	}
	return fx
}

func checkCalls(t *testing.T, input, cleaned string, gotCleaned string, want []toolCallExpect, got []ToolCall) {
	t.Helper()
	if gotCleaned != cleaned {
		t.Errorf("cleaned text for %q\n got  %q\n want %q", input, gotCleaned, cleaned)
	}
	if len(got) != len(want) {
		t.Fatalf("call count for %q: got %d, want %d", input, len(got), len(want))
	}
	for i := range want {
		if got[i].Function.Name != want[i].Name {
			t.Errorf("call %d name for %q: got %q, want %q",
				i, input, got[i].Function.Name, want[i].Name)
		}
		if got[i].Function.Arguments != want[i].Arguments {
			t.Errorf("call %d arguments for %q\n got  %q\n want %q",
				i, input, got[i].Function.Arguments, want[i].Arguments)
		}
		if got[i].Type != "function" {
			t.Errorf("call %d type for %q: got %q, want %q", i, input, got[i].Type, "function")
		}
		if got[i].ID == "" {
			t.Errorf("call %d for %q has empty id", i, input)
		}
	}
}

func TestParseXMLToolCallsParity(t *testing.T) {
	for _, fx := range loadToolCallFixtures(t).XML {
		t.Run(fx.Input, func(t *testing.T) {
			cleaned, calls := ParseXMLToolCalls(fx.Input)
			checkCalls(t, fx.Input, fx.Cleaned, cleaned, fx.Calls, calls)
		})
	}
}

func TestParseNamespacedToolCallsParity(t *testing.T) {
	for _, fx := range loadToolCallFixtures(t).Namespaced {
		t.Run(fx.Input, func(t *testing.T) {
			cleaned, calls := ParseNamespacedToolCalls(fx.Input, fx.Namespace)
			checkCalls(t, fx.Input, fx.Cleaned, cleaned, fx.Calls, calls)
		})
	}
}

func TestParseBracketToolCallsParity(t *testing.T) {
	for _, fx := range loadToolCallFixtures(t).Bracket {
		t.Run(fx.Input, func(t *testing.T) {
			cleaned, calls := ParseBracketToolCalls(fx.Input)
			checkCalls(t, fx.Input, fx.Cleaned, cleaned, fx.Calls, calls)
		})
	}
}

// TestPythonQuoteEscaping pins the escaping rules against Python's json.dumps
// with ensure_ascii=False: named control escapes, \u00xx for other controls,
// and non-ASCII passed through unchanged.
func TestPythonQuoteEscaping(t *testing.T) {
	cases := []struct{ in, want string }{
		{"plain", `"plain"`},
		{"a\"b", `"a\"b"`},
		{"a\\b", `"a\\b"`},
		{"line1\nline2", `"line1\nline2"`},
		{"tab\there", `"tab\there"`},
		{"\x01", "\"\\u0001\""},
		{"café", `"café"`},
		{"日本語", `"日本語"`},
	}
	for _, c := range cases {
		v := jval{kind: kindString, s: c.in}
		if got := v.dump(); got != c.want {
			t.Errorf("dump(%q) = %s, want %s", c.in, got, c.want)
		}
	}
}

func BenchmarkParseXMLToolCalls(b *testing.B) {
	const in = `here you go <tool_call>{"name": "get_weather", ` +
		`"arguments": {"city": "Paris", "units": "metric", "days": 3}}</tool_call> done`
	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseXMLToolCalls(in)
	}
}

func BenchmarkParseBracketToolCalls(b *testing.B) {
	const in = `[Calling tool: search({"q": "cats", "limit": 5})] and [Tool call: ping]`
	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseBracketToolCalls(in)
	}
}
