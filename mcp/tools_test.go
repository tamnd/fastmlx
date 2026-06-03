// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type toolSpec struct {
	ServerName  string          `json:"server_name"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

func (s toolSpec) tool() Tool {
	return Tool{ServerName: s.ServerName, Name: s.Name, Description: s.Description, InputSchema: s.InputSchema}
}

type mcpToolsFixture struct {
	ToolToOpenAI []struct {
		Tool toolSpec        `json:"tool"`
		Out  json.RawMessage `json:"out"`
	} `json:"tool_to_openai"`
	OpenAICallToMCP []struct {
		ToolCall  json.RawMessage `json:"tool_call"`
		Server    string          `json:"server"`
		Tool      string          `json:"tool"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"openai_call_to_mcp"`
	MergeTools []struct {
		MCP  []toolSpec        `json:"mcp"`
		User []json.RawMessage `json:"user"`
		Out  json.RawMessage   `json:"out"`
	} `json:"merge_tools"`
	Extract []struct {
		Response  json.RawMessage `json:"response"`
		ToolCalls json.RawMessage `json:"tool_calls"`
		Has       bool            `json:"has"`
	} `json:"extract"`
}

func loadMCPTools(t *testing.T) mcpToolsFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/mcptools.json")
	if err != nil {
		t.Fatal(err)
	}
	var f mcpToolsFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// normalize marshals a Go value and decodes it back to a generic shape so two
// values compare structurally, independent of map key order and int/float form.
func normalize(t *testing.T, v any) any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func decodeAny(t *testing.T, raw json.RawMessage) any {
	t.Helper()
	if raw == nil {
		return nil
	}
	var out any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestToolToOpenAIParity(t *testing.T) {
	for i, c := range loadMCPTools(t).ToolToOpenAI {
		got := normalize(t, ToolToOpenAI(c.Tool.tool()))
		want := decodeAny(t, c.Out)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("ToolToOpenAI case %d:\n got  %v\n want %v", i, got, want)
		}
	}
}

func TestOpenAICallToMCPParity(t *testing.T) {
	for i, c := range loadMCPTools(t).OpenAICallToMCP {
		var call map[string]any
		if err := json.Unmarshal(c.ToolCall, &call); err != nil {
			t.Fatal(err)
		}
		server, tool, args := OpenAICallToMCP(call)
		if server != c.Server || tool != c.Tool {
			t.Errorf("case %d: got (%q,%q) want (%q,%q)", i, server, tool, c.Server, c.Tool)
		}
		gotArgs := normalize(t, args)
		wantArgs := decodeAny(t, c.Arguments)
		if !reflect.DeepEqual(gotArgs, wantArgs) {
			t.Errorf("case %d arguments:\n got  %v\n want %v", i, gotArgs, wantArgs)
		}
	}
}

func TestMergeToolsParity(t *testing.T) {
	for i, c := range loadMCPTools(t).MergeTools {
		mcpTools := make([]Tool, len(c.MCP))
		for j, s := range c.MCP {
			mcpTools[j] = s.tool()
		}
		var user []map[string]any
		for _, raw := range c.User {
			var m map[string]any
			if err := json.Unmarshal(raw, &m); err != nil {
				t.Fatal(err)
			}
			user = append(user, m)
		}
		got := normalize(t, MergeTools(mcpTools, user))
		want := decodeAny(t, c.Out)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("MergeTools case %d:\n got  %v\n want %v", i, got, want)
		}
	}
}

func TestExtractAndHasToolCallsParity(t *testing.T) {
	for i, c := range loadMCPTools(t).Extract {
		var resp map[string]any
		if err := json.Unmarshal(c.Response, &resp); err != nil {
			t.Fatal(err)
		}
		got := normalize(t, ExtractToolCalls(resp))
		want := decodeAny(t, c.ToolCalls)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("ExtractToolCalls case %d:\n got  %v\n want %v", i, got, want)
		}
		if h := HasToolCalls(resp); h != c.Has {
			t.Errorf("HasToolCalls case %d: got %v want %v", i, h, c.Has)
		}
	}
}

func BenchmarkToolToOpenAI(b *testing.B) {
	tool := Tool{ServerName: "srv", Name: "search", Description: "Search", InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`)}
	b.ReportAllocs()
	for b.Loop() {
		_ = ToolToOpenAI(tool)
	}
}

func BenchmarkOpenAICallToMCP(b *testing.B) {
	call := map[string]any{"function": map[string]any{"name": "srv__search", "arguments": `{"q":"hello"}`}}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _ = OpenAICallToMCP(call)
	}
}
