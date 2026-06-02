// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

// TestStdioHelperServer is not a real test: when MCP_STDIO_HELPER is set it acts
// as a minimal MCP server over stdio, so the stdio transport can be exercised
// end to end against a real subprocess. It answers initialize, tools/list, and
// tools/call with newline-delimited JSON-RPC.
func TestStdioHelperServer(t *testing.T) {
	if os.Getenv("MCP_STDIO_HELPER") != "1" {
		t.Skip("helper process")
	}
	in := bufio.NewReader(os.Stdin)
	out := json.NewEncoder(os.Stdout)
	for {
		line, err := in.ReadBytes('\n')
		if err != nil && len(line) == 0 {
			return
		}
		var req struct {
			ID     *int64 `json:"id"`
			Method string `json:"method"`
		}
		if json.Unmarshal(line, &req) != nil {
			continue
		}
		if req.ID == nil { // notification
			continue
		}
		var result json.RawMessage
		switch req.Method {
		case "initialize":
			result = json.RawMessage(`{"protocolVersion":"2024-11-05","serverInfo":{"name":"helper"}}`)
		case "tools/list":
			result = json.RawMessage(`{"tools":[{"name":"echo","description":"echo back","inputSchema":{"type":"object"}}]}`)
		case "tools/call":
			result = json.RawMessage(`{"content":[{"type":"text","text":"echoed"}],"isError":false}`)
		default:
			result = json.RawMessage(`{}`)
		}
		_ = out.Encode(map[string]any{"jsonrpc": "2.0", "id": req.ID, "result": result})
	}
}

func TestStdioTransportEndToEnd(t *testing.T) {
	cfg := ServerConfig{
		Name:      "helper",
		Transport: TransportStdio,
		Command:   os.Args[0],
		Args:      []string{"-test.run=TestStdioHelperServer"},
		Env:       map[string]string{"MCP_STDIO_HELPER": "1"},
		Enabled:   true,
		Timeout:   10,
	}
	c := NewClient(cfg)
	ok, err := c.Connect(context.Background())
	if err != nil || !ok {
		t.Fatalf("connect: ok=%v err=%v", ok, err)
	}
	defer c.Disconnect()

	tools := c.Tools()
	if len(tools) != 1 || tools[0].FullName() != "helper__echo" {
		t.Fatalf("tools = %+v", tools)
	}

	res := c.CallTool(context.Background(), "echo", json.RawMessage(`{"msg":"hi"}`), 0)
	if res.IsError || string(res.Content) != `"echoed"` {
		t.Fatalf("call result = %+v", res)
	}
	fmt.Fprintln(os.Stderr, "stdio end-to-end ok")
}
