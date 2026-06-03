// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestGemma4FallbackParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "gemma4_tool_call_fallback.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fixture struct {
		Robust []struct {
			Input  string          `json:"input"`
			Result json.RawMessage `json:"result"`
			Error  *string         `json:"error"`
		} `json:"robust"`
		Fallback []struct {
			Input  string          `json:"input"`
			Result json.RawMessage `json:"result"`
			Error  *string         `json:"error"`
		} `json:"fallback"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range fixture.Robust {
		got, ok := ArgsToJSONRobust(c.Input)
		if c.Error != nil {
			if ok {
				t.Errorf("robust case %d (%q): expected failure, got %s", i, c.Input, got.dump())
			}
			continue
		}
		if !ok {
			t.Errorf("robust case %d (%q): unexpected failure", i, c.Input)
			continue
		}
		want, _ := parseOrdered(string(c.Result))
		if g, w := got.dump(), want.dump(); g != w {
			t.Errorf("robust case %d (%q):\n got %s\nwant %s", i, c.Input, g, w)
		}
	}

	for i, c := range fixture.Fallback {
		got, err := ParseGemma4ToolCallFallback(c.Input)
		if c.Error != nil {
			if err == nil {
				t.Errorf("fallback case %d (%q): expected error, got %s", i, c.Input, got.dump())
				continue
			}
			if err != errNoGemma4Call {
				t.Errorf("fallback case %d (%q): unexpected error %v", i, c.Input, err)
			}
			continue
		}
		if err != nil {
			t.Errorf("fallback case %d (%q): unexpected error %v", i, c.Input, err)
			continue
		}
		want, _ := parseOrdered(string(c.Result))
		if g, w := got.dump(), want.dump(); g != w {
			t.Errorf("fallback case %d (%q):\n got %s\nwant %s", i, c.Input, g, w)
		}
	}
}

func BenchmarkParseGemma4ToolCallFallback(b *testing.B) {
	b.ReportAllocs()
	const text = `call:get_weather{location: Tokyo, unit: celsius} call:lookup{id: 42}`
	for b.Loop() {
		_, _ = ParseGemma4ToolCallFallback(text)
	}
}
