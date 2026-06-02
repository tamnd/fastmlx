// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

// Fixtures in testdata/parity/mcp_tools.json are captured from the reference
// mcp/tools.py conversions. The OpenAI tool definitions and merged tool lists
// feed the chat-template tool path where object key order is not contractual,
// so those comparisons are structural (canonJSON, which preserves array order
// so tool ordering is still checked). The (server, tool, arguments) triple and
// the tool-role message are compared structurally as well.

type mcpToolDescriptor struct {
	ServerName  string          `json:"server_name"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

func (d mcpToolDescriptor) toMCPTool(t *testing.T) MCPTool {
	t.Helper()
	schema := jval{kind: kindObject}
	if len(d.InputSchema) > 0 && string(d.InputSchema) != "null" {
		v, ok := parseOrdered(string(d.InputSchema))
		if !ok {
			t.Fatalf("schema did not parse: %s", d.InputSchema)
		}
		schema = v
	}
	return MCPTool{
		ServerName:  d.ServerName,
		Name:        d.Name,
		Description: d.Description,
		InputSchema: schema,
	}
}

type mcpToolsFixtures struct {
	MCPToolCases []struct {
		Label  string              `json:"label"`
		Tools  []mcpToolDescriptor `json:"tools"`
		Result json.RawMessage     `json:"result"`
	} `json:"mcp_tool_cases"`
	MergeCases []struct {
		Label     string              `json:"label"`
		MCPTools  []mcpToolDescriptor `json:"mcp_tools"`
		UserTools json.RawMessage     `json:"user_tools"`
		Result    json.RawMessage     `json:"result"`
	} `json:"merge_cases"`
	CallCases []struct {
		Label    string          `json:"label"`
		ToolCall json.RawMessage `json:"tool_call"`
		Result   struct {
			Server    string          `json:"server"`
			Tool      string          `json:"tool"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"result"`
	} `json:"call_cases"`
	ResultCases []struct {
		Label  string `json:"label"`
		Result struct {
			ToolName     string          `json:"tool_name"`
			Content      json.RawMessage `json:"content"`
			IsError      bool            `json:"is_error"`
			ErrorMessage *string         `json:"error_message"`
		} `json:"result"`
		CallID  string          `json:"call_id"`
		Message json.RawMessage `json:"message"`
	} `json:"result_cases"`
}

func loadMCPToolsFixtures(t testing.TB) mcpToolsFixtures {
	t.Helper()
	data, err := os.ReadFile("testdata/parity/mcp_tools.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fx mcpToolsFixtures
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("decode fixtures: %v", err)
	}
	return fx
}

func TestMCPToolsToOpenAIParity(t *testing.T) {
	for _, c := range loadMCPToolsFixtures(t).MCPToolCases {
		t.Run(c.Label, func(t *testing.T) {
			tools := make([]MCPTool, 0, len(c.Tools))
			for _, d := range c.Tools {
				tools = append(tools, d.toMCPTool(t))
			}
			got := canonJSON(t, []byte(jval{kind: kindArray, arr: MCPToolsToOpenAI(tools)}.dumpASCII()))
			want := canonJSON(t, c.Result)
			if got != want {
				t.Errorf("case %q\n got  %s\n want %s", c.Label, got, want)
			}
		})
	}
}

func TestMergeToolsParity(t *testing.T) {
	for _, c := range loadMCPToolsFixtures(t).MergeCases {
		t.Run(c.Label, func(t *testing.T) {
			tools := make([]MCPTool, 0, len(c.MCPTools))
			for _, d := range c.MCPTools {
				tools = append(tools, d.toMCPTool(t))
			}
			var userTools []jval
			if len(c.UserTools) > 0 && string(c.UserTools) != "null" {
				v, ok := parseOrdered(string(c.UserTools))
				if !ok || v.kind != kindArray {
					t.Fatalf("user_tools did not parse as array: %s", c.UserTools)
				}
				userTools = v.arr
			}
			got := canonJSON(t, []byte(jval{kind: kindArray, arr: MergeTools(tools, userTools)}.dumpASCII()))
			want := canonJSON(t, c.Result)
			if got != want {
				t.Errorf("case %q\n got  %s\n want %s", c.Label, got, want)
			}
		})
	}
}

func TestOpenAICallToMCPParity(t *testing.T) {
	for _, c := range loadMCPToolsFixtures(t).CallCases {
		t.Run(c.Label, func(t *testing.T) {
			call, _ := parseOrdered(string(c.ToolCall))
			server, tool, args := OpenAICallToMCP(call)
			if server != c.Result.Server {
				t.Errorf("server = %q, want %q", server, c.Result.Server)
			}
			if tool != c.Result.Tool {
				t.Errorf("tool = %q, want %q", tool, c.Result.Tool)
			}
			gotArgs := canonJSON(t, []byte(args.dumpASCII()))
			wantArgs := canonJSON(t, c.Result.Arguments)
			if gotArgs != wantArgs {
				t.Errorf("arguments\n got  %s\n want %s", gotArgs, wantArgs)
			}
		})
	}
}

func TestMCPToolResultToMessageParity(t *testing.T) {
	for _, c := range loadMCPToolsFixtures(t).ResultCases {
		t.Run(c.Label, func(t *testing.T) {
			content := jnull()
			if len(c.Result.Content) > 0 && string(c.Result.Content) != "null" {
				v, ok := parseOrdered(string(c.Result.Content))
				if !ok {
					t.Fatalf("content did not parse: %s", c.Result.Content)
				}
				content = v
			}
			errMsg := ""
			if c.Result.ErrorMessage != nil {
				errMsg = *c.Result.ErrorMessage
			}
			result := MCPToolResult{
				ToolName:     c.Result.ToolName,
				Content:      content,
				IsError:      c.Result.IsError,
				ErrorMessage: errMsg,
			}
			got := canonJSON(t, []byte(result.ToMessage(c.CallID).dumpASCII()))
			want := canonJSON(t, c.Message)
			if got != want {
				t.Errorf("case %q\n got  %s\n want %s", c.Label, got, want)
			}
		})
	}
}

func TestHasToolCalls(t *testing.T) {
	with, _ := parseOrdered(`{"choices":[{"message":{"tool_calls":[{"id":"c1"}]}}]}`)
	if !HasToolCalls(with) {
		t.Error("expected HasToolCalls true for a response with tool calls")
	}
	without, _ := parseOrdered(`{"choices":[{"message":{"content":"hi"}}]}`)
	if HasToolCalls(without) {
		t.Error("expected HasToolCalls false for a response without tool calls")
	}
	empty, _ := parseOrdered(`{"choices":[]}`)
	if HasToolCalls(empty) {
		t.Error("expected HasToolCalls false for empty choices")
	}
}

func BenchmarkMergeTools(b *testing.B) {
	tools := []MCPTool{
		{ServerName: "fs", Name: "read", Description: "Read a file",
			InputSchema: jobj("type", jstr("object"), "properties", jval{kind: kindObject})},
		{ServerName: "git", Name: "status", Description: "Repo status"},
	}
	userTool, _ := parseOrdered(`{"type":"function","function":{"name":"calc","parameters":{}}}`)
	userTools := []jval{userTool}
	b.ReportAllocs()
	for b.Loop() {
		_ = MergeTools(tools, userTools)
	}
}
