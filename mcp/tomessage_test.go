// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"encoding/json"
	"os"
	"testing"
)

type toMessageFixture struct {
	Cases []struct {
		ContentRaw   string          `json:"content_raw"`
		IsError      bool            `json:"is_error"`
		ErrorMessage string          `json:"error_message"`
		ToolCallID   string          `json:"tool_call_id"`
		Out          struct {
			Role       string `json:"role"`
			ToolCallID string `json:"tool_call_id"`
			Content    string `json:"content"`
		} `json:"out"`
	} `json:"cases"`
}

func loadToMessage(t *testing.T) toMessageFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/tomessage.json")
	if err != nil {
		t.Fatal(err)
	}
	var f toMessageFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestToMessageParity(t *testing.T) {
	for i, c := range loadToMessage(t).Cases {
		result := ToolResult{Content: json.RawMessage(c.ContentRaw), IsError: c.IsError, ErrorMessage: c.ErrorMessage}
		msg := FormatToolResult(result, c.ToolCallID)
		if got := msg["role"]; got != c.Out.Role {
			t.Errorf("case %d role: got %v want %v", i, got, c.Out.Role)
		}
		if got := msg["tool_call_id"]; got != c.Out.ToolCallID {
			t.Errorf("case %d tool_call_id: got %v want %v", i, got, c.Out.ToolCallID)
		}
		if got := msg["content"]; got != c.Out.Content {
			t.Errorf("case %d content:\n got  %q\n want %q", i, got, c.Out.Content)
		}
	}
}

func TestFormatToolResultsOrder(t *testing.T) {
	results := []ResultWithCallID{
		{Result: ToolResult{Content: json.RawMessage(`"first"`)}, ToolCallID: "a"},
		{Result: ToolResult{IsError: true, ErrorMessage: "boom"}, ToolCallID: "b"},
		{Result: ToolResult{Content: json.RawMessage(`{"k":1}`)}, ToolCallID: "c"},
	}
	msgs := FormatToolResults(results)
	if len(msgs) != 3 {
		t.Fatalf("got %d messages, want 3", len(msgs))
	}
	wantIDs := []string{"a", "b", "c"}
	wantContent := []string{"first", "Error: boom", `{"k": 1}`}
	for i, m := range msgs {
		if m["tool_call_id"] != wantIDs[i] {
			t.Errorf("msg %d id: got %v want %v", i, m["tool_call_id"], wantIDs[i])
		}
		if m["content"] != wantContent[i] {
			t.Errorf("msg %d content: got %q want %q", i, m["content"], wantContent[i])
		}
	}
}

func BenchmarkToMessageJSON(b *testing.B) {
	result := ToolResult{Content: json.RawMessage(`{"b":2,"a":1,"nested":{"y":true,"items":[1,2,3]}}`)}
	b.ReportAllocs()
	for b.Loop() {
		_ = FormatToolResult(result, "call_1")
	}
}

func BenchmarkPyJSONDumps(b *testing.B) {
	raw := json.RawMessage(`{"msg":"café ☃","arr":[1.5,-2.25,0.0001],"n":1000000}`)
	b.ReportAllocs()
	for b.Loop() {
		_ = pyJSONDumps(raw)
	}
}
