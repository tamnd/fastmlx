// SPDX-License-Identifier: MIT OR Apache-2.0

package routes

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tamnd/fastmlx/mcp"
)

// fakeMCP is an in-memory MCP manager for route tests.
type fakeMCP struct {
	tools    []mcp.Tool
	statuses []mcp.ServerStatus
	result   mcp.ToolResult
	gotName  string
	gotArgs  string
}

func (f *fakeMCP) AllTools() []mcp.Tool               { return f.tools }
func (f *fakeMCP) ServerStatuses() []mcp.ServerStatus { return f.statuses }
func (f *fakeMCP) ExecuteTool(_ context.Context, name string, args json.RawMessage, _ float64) mcp.ToolResult {
	f.gotName = name
	f.gotArgs = string(args)
	return f.result
}

func newReq(method, path, body string) (*http.Request, *httptest.ResponseRecorder) {
	r := httptest.NewRequest(method, path, strings.NewReader(body))
	return r, httptest.NewRecorder()
}

func TestMCPToolsNoManager(t *testing.T) {
	rt := New(nil)
	r, w := newReq("GET", "/v1/mcp/tools", "")
	rt.MCPTools(w, r)
	if w.Code != 200 {
		t.Fatalf("code %d", w.Code)
	}
	if got := w.Body.String(); !strings.Contains(got, `"tools":[]`) || !strings.Contains(got, `"count":0`) {
		t.Errorf("body = %s", got)
	}
}

func TestMCPToolsWithManager(t *testing.T) {
	rt := New(nil)
	rt.SetMCPManager(&fakeMCP{tools: []mcp.Tool{
		{ServerName: "fs", Name: "read", Description: "read a file", InputSchema: json.RawMessage(`{"type":"object"}`)},
		{ServerName: "db", Name: "query", Description: "run a query"},
	}})
	r, w := newReq("GET", "/v1/mcp/tools", "")
	rt.MCPTools(w, r)

	var resp struct {
		Tools []struct {
			Name       string          `json:"name"`
			Server     string          `json:"server"`
			Parameters json.RawMessage `json:"parameters"`
		} `json:"tools"`
		Count int `json:"count"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Count != 2 || resp.Tools[0].Name != "fs__read" || resp.Tools[1].Name != "db__query" {
		t.Fatalf("resp = %+v", resp)
	}
	// A tool with no schema defaults to an empty object.
	if string(resp.Tools[1].Parameters) != "{}" {
		t.Errorf("default parameters = %s", resp.Tools[1].Parameters)
	}
}

func TestMCPServers(t *testing.T) {
	rt := New(nil)
	rt.SetMCPManager(&fakeMCP{statuses: []mcp.ServerStatus{
		{Name: "fs", State: mcp.StateConnected, Transport: mcp.TransportStdio, ToolsCount: 3},
		{Name: "db", State: mcp.StateError, Transport: mcp.TransportSSE, Error: "refused"},
	}})
	r, w := newReq("GET", "/v1/mcp/servers", "")
	rt.MCPServers(w, r)

	var resp struct {
		Servers []struct {
			Name       string  `json:"name"`
			State      string  `json:"state"`
			Transport  string  `json:"transport"`
			ToolsCount int     `json:"tools_count"`
			Error      *string `json:"error"`
		} `json:"servers"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Servers) != 2 {
		t.Fatalf("servers = %d", len(resp.Servers))
	}
	if resp.Servers[0].State != "connected" || resp.Servers[0].Error != nil {
		t.Errorf("server[0] = %+v", resp.Servers[0])
	}
	if resp.Servers[1].State != "error" || resp.Servers[1].Transport != "sse" || resp.Servers[1].Error == nil || *resp.Servers[1].Error != "refused" {
		t.Errorf("server[1] = %+v", resp.Servers[1])
	}
}

func TestMCPExecuteNoManager(t *testing.T) {
	rt := New(nil)
	r, w := newReq("POST", "/v1/mcp/execute", `{"tool_name":"x","arguments":{}}`)
	rt.MCPExecute(w, r)
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("code %d", w.Code)
	}
	if !strings.Contains(w.Body.String(), "MCP not configured") {
		t.Errorf("body = %s", w.Body.String())
	}
}

func TestMCPExecuteSuccess(t *testing.T) {
	f := &fakeMCP{result: mcp.ToolResult{ToolName: "fs__read", Content: json.RawMessage(`"file data"`)}}
	rt := New(nil)
	rt.SetMCPManager(f)
	r, w := newReq("POST", "/v1/mcp/execute", `{"tool_name":"fs__read","arguments":{"path":"/x"}}`)
	rt.MCPExecute(w, r)
	if w.Code != 200 {
		t.Fatalf("code %d: %s", w.Code, w.Body.String())
	}
	if f.gotName != "fs__read" || f.gotArgs != `{"path":"/x"}` {
		t.Errorf("forwarded name=%q args=%q", f.gotName, f.gotArgs)
	}
	var resp struct {
		ToolName     string          `json:"tool_name"`
		Content      json.RawMessage `json:"content"`
		IsError      bool            `json:"is_error"`
		ErrorMessage *string         `json:"error_message"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.ToolName != "fs__read" || string(resp.Content) != `"file data"` || resp.IsError || resp.ErrorMessage != nil {
		t.Errorf("resp = %+v", resp)
	}
}

func TestMCPExecuteToolAlias(t *testing.T) {
	f := &fakeMCP{result: mcp.ToolResult{ToolName: "x"}}
	rt := New(nil)
	rt.SetMCPManager(f)
	// "tool" is accepted in place of "tool_name".
	r, w := newReq("POST", "/v1/mcp/execute", `{"tool":"x"}`)
	rt.MCPExecute(w, r)
	if f.gotName != "x" {
		t.Errorf("alias not honored: %q", f.gotName)
	}
}

func TestMCPExecuteError(t *testing.T) {
	f := &fakeMCP{result: mcp.ToolResult{ToolName: "x", IsError: true, ErrorMessage: "boom"}}
	rt := New(nil)
	rt.SetMCPManager(f)
	r, w := newReq("POST", "/v1/mcp/execute", `{"tool_name":"x"}`)
	rt.MCPExecute(w, r)
	var resp struct {
		Content      json.RawMessage `json:"content"`
		IsError      bool            `json:"is_error"`
		ErrorMessage *string         `json:"error_message"`
	}
	json.Unmarshal(w.Body.Bytes(), &resp)
	if !resp.IsError || resp.ErrorMessage == nil || *resp.ErrorMessage != "boom" || string(resp.Content) != "null" {
		t.Errorf("resp = %+v", resp)
	}
}
