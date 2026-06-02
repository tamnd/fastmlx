// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"
)

// fakeTransport answers the MCP handshake and tool calls in-process so the
// client lifecycle can be tested without a subprocess.
type fakeTransport struct {
	tools      string // JSON for the tools/list result
	callResult string // JSON for the tools/call result
	callErr    error
	blockCall  bool // block tools/call until the context is cancelled
	initErr    error
	closed     bool
	lastName   string
	lastArgs   string
}

func (f *fakeTransport) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	switch method {
	case "initialize":
		if f.initErr != nil {
			return nil, f.initErr
		}
		return json.RawMessage(`{"protocolVersion":"2024-11-05","serverInfo":{"name":"fake"}}`), nil
	case "tools/list":
		if f.tools == "" {
			return json.RawMessage(`{"tools":[]}`), nil
		}
		return json.RawMessage(f.tools), nil
	case "tools/call":
		if f.blockCall {
			<-ctx.Done()
			return nil, ctx.Err()
		}
		if f.callErr != nil {
			return nil, f.callErr
		}
		if m, ok := params.(map[string]any); ok {
			f.lastName, _ = m["name"].(string)
			if a, ok := m["arguments"].(json.RawMessage); ok {
				f.lastArgs = string(a)
			}
		}
		return json.RawMessage(f.callResult), nil
	}
	return json.RawMessage(`{}`), nil
}

func (f *fakeTransport) notify(ctx context.Context, method string, params any) error { return nil }
func (f *fakeTransport) close() error                                                { f.closed = true; return nil }

func newFakeClient(cfg ServerConfig, tr *fakeTransport) *Client {
	c := NewClient(cfg)
	c.dial = func(ctx context.Context, _ ServerConfig) (transport, error) {
		return tr, nil
	}
	return c
}

func TestClientConnectDiscoversTools(t *testing.T) {
	tr := &fakeTransport{tools: `{"tools":[
		{"name":"read","description":"read a file","inputSchema":{"type":"object","properties":{"path":{"type":"string"}}}},
		{"name":"write","description":"write a file","inputSchema":{}}
	]}`}
	c := newFakeClient(ServerConfig{Name: "fs", Transport: TransportStdio, Command: "x", Enabled: true, Timeout: 5}, tr)

	ok, err := c.Connect(context.Background())
	if err != nil || !ok {
		t.Fatalf("connect: ok=%v err=%v", ok, err)
	}
	if !c.IsConnected() {
		t.Fatal("not connected")
	}
	tools := c.Tools()
	if len(tools) != 2 {
		t.Fatalf("tools = %d", len(tools))
	}
	if tools[0].FullName() != "fs__read" || tools[1].FullName() != "fs__write" {
		t.Errorf("names = %q,%q", tools[0].FullName(), tools[1].FullName())
	}
	st := c.Status()
	if st.State != StateConnected || st.ToolsCount != 2 || st.Transport != TransportStdio {
		t.Errorf("status = %+v", st)
	}
}

func TestClientConnectDisabled(t *testing.T) {
	c := newFakeClient(ServerConfig{Name: "off", Transport: TransportStdio, Command: "x", Enabled: false}, &fakeTransport{})
	ok, err := c.Connect(context.Background())
	if ok || err != nil {
		t.Fatalf("disabled connect: ok=%v err=%v", ok, err)
	}
	if c.IsConnected() {
		t.Error("disabled server should not connect")
	}
}

func TestClientInitializeErrorSetsErrorState(t *testing.T) {
	tr := &fakeTransport{initErr: fmt.Errorf("handshake refused")}
	c := newFakeClient(ServerConfig{Name: "bad", Transport: TransportStdio, Command: "x", Enabled: true}, tr)
	ok, err := c.Connect(context.Background())
	if ok || err == nil {
		t.Fatalf("expected failure, got ok=%v err=%v", ok, err)
	}
	if c.State() != StateError {
		t.Errorf("state = %v", c.State())
	}
	if !tr.closed {
		t.Error("transport should be cleaned up on failure")
	}
}

func TestClientCallToolSingleText(t *testing.T) {
	tr := &fakeTransport{callResult: `{"content":[{"type":"text","text":"hello world"}],"isError":false}`}
	c := newFakeClient(ServerConfig{Name: "fs", Transport: TransportStdio, Command: "x", Enabled: true, Timeout: 5}, tr)
	c.Connect(context.Background())

	res := c.CallTool(context.Background(), "read", json.RawMessage(`{"path":"/x"}`), 0)
	if res.IsError {
		t.Fatalf("unexpected error: %s", res.ErrorMessage)
	}
	if string(res.Content) != `"hello world"` {
		t.Errorf("content = %s", res.Content)
	}
	if tr.lastName != "read" || tr.lastArgs != `{"path":"/x"}` {
		t.Errorf("forwarded name=%q args=%q", tr.lastName, tr.lastArgs)
	}
}

func TestClientCallToolMultipleItems(t *testing.T) {
	tr := &fakeTransport{callResult: `{"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}`}
	c := newFakeClient(ServerConfig{Name: "fs", Transport: TransportStdio, Command: "x", Enabled: true, Timeout: 5}, tr)
	c.Connect(context.Background())

	res := c.CallTool(context.Background(), "read", nil, 0)
	if string(res.Content) != `["a","b"]` {
		t.Errorf("content = %s", res.Content)
	}
}

func TestClientCallToolStructuredFallback(t *testing.T) {
	tr := &fakeTransport{callResult: `{"content":[],"structuredContent":{"rows":3}}`}
	c := newFakeClient(ServerConfig{Name: "db", Transport: TransportStdio, Command: "x", Enabled: true, Timeout: 5}, tr)
	c.Connect(context.Background())

	res := c.CallTool(context.Background(), "query", nil, 0)
	if string(res.Content) != `{"rows":3}` {
		t.Errorf("content = %s", res.Content)
	}
}

func TestClientCallToolNotConnected(t *testing.T) {
	c := newFakeClient(ServerConfig{Name: "fs", Transport: TransportStdio, Command: "x", Enabled: true}, &fakeTransport{})
	res := c.CallTool(context.Background(), "read", nil, 0)
	if !res.IsError || res.ErrorMessage != "Not connected to server 'fs'" {
		t.Errorf("result = %+v", res)
	}
}

func TestClientCallToolTimeout(t *testing.T) {
	tr := &fakeTransport{blockCall: true}
	c := newFakeClient(ServerConfig{Name: "fs", Transport: TransportStdio, Command: "x", Enabled: true, Timeout: 5}, tr)
	c.Connect(context.Background())

	start := time.Now()
	res := c.CallTool(context.Background(), "slow", nil, 0.05)
	if time.Since(start) > time.Second {
		t.Fatal("timeout did not fire promptly")
	}
	if !res.IsError || res.ErrorMessage != "Tool call timed out after 0.05s" {
		t.Errorf("result = %+v", res)
	}
}

func TestClientDisconnect(t *testing.T) {
	tr := &fakeTransport{}
	c := newFakeClient(ServerConfig{Name: "fs", Transport: TransportStdio, Command: "x", Enabled: true, Timeout: 5}, tr)
	c.Connect(context.Background())
	c.Disconnect()
	if c.State() != StateDisconnected || len(c.Tools()) != 0 || !tr.closed {
		t.Errorf("after disconnect: state=%v tools=%d closed=%v", c.State(), len(c.Tools()), tr.closed)
	}
}

func TestDialRealUnsupportedTransports(t *testing.T) {
	for _, tp := range []Transport{TransportSSE, TransportStreamableHTTP} {
		_, err := dialReal(context.Background(), ServerConfig{Name: "x", Transport: tp, URL: "http://x"})
		if err == nil {
			t.Errorf("%s: expected not-implemented error", tp)
		}
	}
}
