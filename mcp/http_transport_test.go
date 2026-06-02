// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func TestScanSSE(t *testing.T) {
	stream := "event: endpoint\ndata: /messages?id=1\n\n" +
		": a comment line\nevent: message\ndata: line one\ndata: line two\n\n"
	r := bufio.NewReader(strings.NewReader(stream))

	first, err := scanSSE(r)
	if err != nil {
		t.Fatal(err)
	}
	if first.event != "endpoint" || first.data != "/messages?id=1" {
		t.Errorf("first = %+v", first)
	}
	second, err := scanSSE(r)
	if err != nil {
		t.Fatal(err)
	}
	if second.event != "message" || second.data != "line one\nline two" {
		t.Errorf("second = %+v", second)
	}
	if _, err := scanSSE(r); err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
}

// rpcReply computes the canned JSON-RPC result for a method, mirroring a tiny
// MCP server. It returns the raw result object (without the envelope).
func rpcReply(method string) string {
	switch method {
	case "initialize":
		return `{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"test","version":"1"}}`
	case "tools/list":
		return `{"tools":[{"name":"echo","description":"echo back","inputSchema":{"type":"object"}}]}`
	case "tools/call":
		return `{"content":[{"type":"text","text":"pong"}]}`
	default:
		return `{}`
	}
}

// TestStreamableHTTPEndToEnd drives a client through dialReal against a
// streamable-http server returning JSON bodies, and checks the session id handed
// back on initialize is echoed on later requests.
func TestStreamableHTTPEndToEnd(t *testing.T) {
	var mu sync.Mutex
	var sessionHeaders []string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req rpcRequest
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)

		mu.Lock()
		sessionHeaders = append(sessionHeaders, r.Header.Get("Mcp-Session-Id"))
		mu.Unlock()

		if req.ID == nil { // a notification
			w.WriteHeader(http.StatusAccepted)
			return
		}
		if req.Method == "initialize" {
			w.Header().Set("Mcp-Session-Id", "sess-123")
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%d,"result":%s}`, *req.ID, rpcReply(req.Method))
	}))
	defer srv.Close()

	c := NewClient(ServerConfig{
		Name: "http", Transport: TransportStreamableHTTP, URL: srv.URL, Enabled: true, Timeout: 5,
	})
	ok, err := c.Connect(context.Background())
	if err != nil || !ok {
		t.Fatalf("connect: ok=%v err=%v", ok, err)
	}
	defer c.Disconnect()

	tools := c.Tools()
	if len(tools) != 1 || tools[0].FullName() != "http__echo" {
		t.Fatalf("tools = %+v", tools)
	}
	res := c.CallTool(context.Background(), "echo", json.RawMessage(`{"x":1}`), 0)
	if res.IsError || string(res.Content) != `"pong"` {
		t.Fatalf("call result = %+v", res)
	}

	// The first request (initialize) carries no session id; everything after the
	// initialize response carries "sess-123".
	mu.Lock()
	defer mu.Unlock()
	if sessionHeaders[0] != "" {
		t.Errorf("initialize should not carry a session id, got %q", sessionHeaders[0])
	}
	for i, h := range sessionHeaders[1:] {
		if h != "sess-123" {
			t.Errorf("request %d session id = %q, want sess-123", i+1, h)
		}
	}
}

// TestStreamableHTTPSSEResponse exercises the branch where the server answers a
// request with a one-shot text/event-stream rather than a JSON body.
func TestStreamableHTTPSSEResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req rpcRequest
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		if req.ID == nil {
			w.WriteHeader(http.StatusAccepted)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		// A server may interleave an unrelated notification before the response.
		fmt.Fprint(w, "event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"ping\"}\n\n")
		fmt.Fprintf(w, "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":%d,\"result\":%s}\n\n",
			*req.ID, rpcReply(req.Method))
	}))
	defer srv.Close()

	tr, err := dialStreamableHTTP(context.Background(), ServerConfig{URL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	defer tr.close()

	raw, err := tr.call(context.Background(), "tools/list", map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	var res struct {
		Tools []struct{ Name string } `json:"tools"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		t.Fatal(err)
	}
	if len(res.Tools) != 1 || res.Tools[0].Name != "echo" {
		t.Errorf("tools = %+v", res.Tools)
	}
}

// sseTestServer is a minimal HTTP+SSE MCP server: a GET stream that first
// advertises the POST endpoint, then relays response messages produced by the
// POST handler.
func sseTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	out := make(chan string, 8)
	mux := http.NewServeMux()
	mux.HandleFunc("/sse", func(w http.ResponseWriter, r *http.Request) {
		fl, ok := w.(http.Flusher)
		if !ok {
			t.Error("response writer is not a flusher")
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "event: endpoint\ndata: /messages\n\n")
		fl.Flush()
		for {
			select {
			case <-r.Context().Done():
				return
			case msg := <-out:
				fmt.Fprintf(w, "event: message\ndata: %s\n\n", msg)
				fl.Flush()
			}
		}
	})
	mux.HandleFunc("/messages", func(w http.ResponseWriter, r *http.Request) {
		var req rpcRequest
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		w.WriteHeader(http.StatusAccepted)
		if req.ID == nil { // notification: no reply
			return
		}
		out <- fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"result":%s}`, *req.ID, rpcReply(req.Method))
	})
	return httptest.NewServer(mux)
}

func TestSSEEndToEnd(t *testing.T) {
	srv := sseTestServer(t)
	defer srv.Close()

	c := NewClient(ServerConfig{
		Name: "sse", Transport: TransportSSE, URL: srv.URL + "/sse", Enabled: true, Timeout: 5,
	})
	ok, err := c.Connect(context.Background())
	if err != nil || !ok {
		t.Fatalf("connect: ok=%v err=%v", ok, err)
	}
	defer c.Disconnect()

	tools := c.Tools()
	if len(tools) != 1 || tools[0].FullName() != "sse__echo" {
		t.Fatalf("tools = %+v", tools)
	}
	res := c.CallTool(context.Background(), "echo", json.RawMessage(`{}`), 0)
	if res.IsError || string(res.Content) != `"pong"` {
		t.Fatalf("call result = %+v", res)
	}
}

func BenchmarkStreamableHTTPCall(b *testing.B) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req rpcRequest
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%d,"result":%s}`, *req.ID, rpcReply("tools/call"))
	}))
	defer srv.Close()

	tr, _ := dialStreamableHTTP(context.Background(), ServerConfig{URL: srv.URL})
	defer tr.close()
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		_, _ = tr.call(ctx, "tools/call", map[string]any{"name": "echo"})
	}
}

func TestDialHTTPTransportsRequireURL(t *testing.T) {
	if _, err := dialStreamableHTTP(context.Background(), ServerConfig{}); err == nil {
		t.Error("streamable-http should require a url")
	}
	if _, err := dialSSE(context.Background(), ServerConfig{}); err == nil {
		t.Error("sse should require a url")
	}
}
