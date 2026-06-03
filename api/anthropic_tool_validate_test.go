// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestValidateAnthropicToolParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "anthropic_tool_validate.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Tool  json.RawMessage `json:"tool"`
		Valid bool            `json:"valid"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		var tool AnthropicTool
		if err := json.Unmarshal(c.Tool, &tool); err != nil {
			t.Fatalf("case %d: decode tool: %v", i, err)
		}
		got := ValidateAnthropicTool(tool)
		if c.Valid && got != nil {
			t.Errorf("case %d (%s): want valid, got error %v", i, tool.Name, got)
		}
		if !c.Valid && got == nil {
			t.Errorf("case %d (%s): want error, got valid", i, tool.Name)
		}
	}
}

func TestValidateAnthropicToolsFirstFailure(t *testing.T) {
	good := AnthropicTool{Name: "ws", Type: new("web_search_20250305")}
	bad := AnthropicTool{Name: "neither"}
	if err := ValidateAnthropicTools([]AnthropicTool{good, good}); err != nil {
		t.Errorf("all-valid batch: unexpected error %v", err)
	}
	if err := ValidateAnthropicTools([]AnthropicTool{good, bad, good}); err == nil {
		t.Errorf("batch with invalid tool: want error, got nil")
	}
	if err := ValidateAnthropicTools(nil); err != nil {
		t.Errorf("empty batch: unexpected error %v", err)
	}
}

func BenchmarkValidateAnthropicTool(b *testing.B) {
	b.ReportAllocs()
	tool := AnthropicTool{Name: "calc", InputSchema: json.RawMessage(`{"type":"object"}`)}
	for b.Loop() {
		_ = ValidateAnthropicTool(tool)
	}
}
