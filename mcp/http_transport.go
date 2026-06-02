// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
)

// The reference delegates the SSE and streamable-http transports to the Python
// MCP SDK plus httpx. This port carries no third-party dependencies, so both
// transports are implemented directly here on net/http against the MCP HTTP
// transport spec: streamable-http is the single-endpoint request/response form,
// and sse is the older two-channel HTTP+SSE form (a long-lived GET stream for
// server messages plus a POST endpoint advertised by an "endpoint" event).

// sseEvent is one parsed server-sent event. Multiple data lines are joined with
// newlines, per the SSE spec.
type sseEvent struct {
	event string
	data  string
}

// scanSSE reads one server-sent event from r, blocking until a blank line ends
// the event or the stream closes. Comment lines (starting with ":") are ignored.
func scanSSE(r *bufio.Reader) (sseEvent, error) {
	var ev sseEvent
	var data []string
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			if len(strings.TrimRight(line, "\r\n")) == 0 && len(data) == 0 {
				return sseEvent{}, err
			}
			// A final event without a trailing blank line still dispatches.
			ev.data = strings.Join(data, "\n")
			return ev, nil
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			ev.data = strings.Join(data, "\n")
			return ev, nil
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		field, val, _ := strings.Cut(line, ":")
		val = strings.TrimPrefix(val, " ")
		switch field {
		case "event":
			ev.event = val
		case "data":
			data = append(data, val)
		}
	}
}

// streamableHTTPTransport speaks JSON-RPC 2.0 over the streamable-http transport:
// every message is POSTed to a single endpoint, and the reply comes back either
// as a JSON body or as a one-shot text/event-stream. A session id handed back on
// the initialize response is echoed on every later request.
type streamableHTTPTransport struct {
	url     string
	headers map[string]string
	client  *http.Client

	mu        sync.Mutex
	sessionID string
	next      int64
}

// dialStreamableHTTP builds a streamable-http transport. The connection itself
// is established lazily on the first request, so there is no upfront dial.
func dialStreamableHTTP(_ context.Context, cfg ServerConfig) (transport, error) {
	if cfg.URL == "" {
		return nil, fmt.Errorf("streamable-http transport requires a url")
	}
	return &streamableHTTPTransport{
		url:     cfg.URL,
		headers: cfg.Headers,
		client:  &http.Client{},
	}, nil
}

func (t *streamableHTTPTransport) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	t.mu.Lock()
	t.next++
	id := t.next
	t.mu.Unlock()
	return t.send(ctx, rpcRequest{JSONRPC: "2.0", ID: &id, Method: method, Params: params}, &id)
}

func (t *streamableHTTPTransport) notify(ctx context.Context, method string, params any) error {
	_, err := t.send(ctx, rpcRequest{JSONRPC: "2.0", Method: method, Params: params}, nil)
	return err
}

// send POSTs one message and, for a request (wantID non-nil), returns its result.
func (t *streamableHTTPTransport) send(ctx context.Context, msg rpcRequest, wantID *int64) (json.RawMessage, error) {
	body, err := json.Marshal(msg)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}
	t.mu.Lock()
	sid := t.sessionID
	t.mu.Unlock()
	if sid != "" {
		req.Header.Set("Mcp-Session-Id", sid)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if s := resp.Header.Get("Mcp-Session-Id"); s != "" {
		t.mu.Lock()
		t.sessionID = s
		t.mu.Unlock()
	}

	if wantID == nil {
		// A notification expects no JSON-RPC reply.
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil, nil
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("http %d from MCP server", resp.StatusCode)
	}

	if strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		return readRPCFromSSE(bufio.NewReader(resp.Body), *wantID)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return matchRPCResult(data, *wantID)
}

func (t *streamableHTTPTransport) close() error { return nil }

// readRPCFromSSE reads an event-stream until it yields the JSON-RPC response with
// the wanted id, skipping any server-initiated requests or notifications.
func readRPCFromSSE(r *bufio.Reader, wantID int64) (json.RawMessage, error) {
	for {
		ev, err := scanSSE(r)
		if err != nil {
			return nil, fmt.Errorf("event stream ended before response: %w", err)
		}
		if ev.data == "" {
			continue
		}
		result, matched, rerr := matchRPCResultOptional([]byte(ev.data), wantID)
		if rerr != nil {
			return nil, rerr
		}
		if matched {
			return result, nil
		}
	}
}

// matchRPCResult parses a single JSON-RPC response and requires its id to match.
func matchRPCResult(data []byte, wantID int64) (json.RawMessage, error) {
	result, matched, err := matchRPCResultOptional(data, wantID)
	if err != nil {
		return nil, err
	}
	if !matched {
		return nil, fmt.Errorf("no JSON-RPC response with id %d", wantID)
	}
	return result, nil
}

// matchRPCResultOptional parses one JSON-RPC message; matched is false when the
// id does not match (a server request, notification, or a different response).
func matchRPCResultOptional(data []byte, wantID int64) (json.RawMessage, bool, error) {
	var resp rpcResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, false, err
	}
	if resp.ID == nil || *resp.ID != wantID {
		return nil, false, nil
	}
	if resp.Error != nil {
		return nil, false, resp.Error
	}
	return resp.Result, true, nil
}

// sseTransport speaks JSON-RPC 2.0 over the older HTTP+SSE transport: a long-lived
// GET stream carries server messages, and the server's first "endpoint" event
// names the URL to POST client messages to. Responses are matched to pending
// calls by id.
type sseTransport struct {
	client  *http.Client
	headers map[string]string
	base    *url.URL

	cancel context.CancelFunc
	body   io.ReadCloser
	closed chan struct{} // closed when the read loop exits

	mu         sync.Mutex
	next       int64
	endpoint   string
	endpointCh chan struct{} // closed once the endpoint event arrives
	pending    map[int64]chan rpcResponse
	connErr    error
}

// dialSSE opens the GET event stream and starts the background reader. The
// returned transport is usable once the server's endpoint event arrives, which
// callers wait for in call/notify.
func dialSSE(_ context.Context, cfg ServerConfig) (transport, error) {
	if cfg.URL == "" {
		return nil, fmt.Errorf("sse transport requires a url")
	}
	base, err := url.Parse(cfg.URL)
	if err != nil {
		return nil, err
	}

	// The stream lives for the connection's lifetime, independent of the dial
	// call's context; close() cancels it.
	connCtx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(connCtx, http.MethodGet, cfg.URL, nil)
	if err != nil {
		cancel()
		return nil, err
	}
	req.Header.Set("Accept", "text/event-stream")
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}
	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		cancel()
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		resp.Body.Close()
		cancel()
		return nil, fmt.Errorf("http %d opening SSE stream", resp.StatusCode)
	}

	t := &sseTransport{
		client:     &http.Client{},
		headers:    cfg.Headers,
		base:       base,
		cancel:     cancel,
		body:       resp.Body,
		closed:     make(chan struct{}),
		endpointCh: make(chan struct{}),
		pending:    make(map[int64]chan rpcResponse),
	}
	go t.readLoop(bufio.NewReader(resp.Body))
	return t, nil
}

// readLoop consumes the event stream, recording the POST endpoint and routing
// responses to waiting callers by id.
func (t *sseTransport) readLoop(r *bufio.Reader) {
	defer close(t.closed)
	for {
		ev, err := scanSSE(r)
		if err != nil {
			t.mu.Lock()
			if t.connErr == nil {
				t.connErr = err
			}
			t.mu.Unlock()
			return
		}
		switch ev.event {
		case "endpoint":
			ref, perr := url.Parse(strings.TrimSpace(ev.data))
			if perr != nil {
				continue
			}
			t.mu.Lock()
			if t.endpoint == "" {
				t.endpoint = t.base.ResolveReference(ref).String()
				close(t.endpointCh)
			}
			t.mu.Unlock()
		default:
			// "message" or an unnamed event carrying a JSON-RPC payload.
			if ev.data == "" {
				continue
			}
			var resp rpcResponse
			if json.Unmarshal([]byte(ev.data), &resp) != nil || resp.ID == nil {
				continue
			}
			t.mu.Lock()
			ch := t.pending[*resp.ID]
			t.mu.Unlock()
			if ch != nil {
				ch <- resp
			}
		}
	}
}

// waitEndpoint blocks until the endpoint event arrives, the context is done, or
// the connection fails.
func (t *sseTransport) waitEndpoint(ctx context.Context) error {
	select {
	case <-t.endpointCh:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-t.closed:
		return t.connError()
	}
}

func (t *sseTransport) connError() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.connErr != nil {
		return t.connErr
	}
	return fmt.Errorf("SSE connection closed")
}

func (t *sseTransport) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	if err := t.waitEndpoint(ctx); err != nil {
		return nil, err
	}
	t.mu.Lock()
	t.next++
	id := t.next
	ch := make(chan rpcResponse, 1)
	t.pending[id] = ch
	endpoint := t.endpoint
	t.mu.Unlock()
	defer func() {
		t.mu.Lock()
		delete(t.pending, id)
		t.mu.Unlock()
	}()

	if err := t.post(ctx, endpoint, rpcRequest{JSONRPC: "2.0", ID: &id, Method: method, Params: params}); err != nil {
		return nil, err
	}
	select {
	case resp := <-ch:
		if resp.Error != nil {
			return nil, resp.Error
		}
		return resp.Result, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-t.closed:
		return nil, t.connError()
	}
}

func (t *sseTransport) notify(ctx context.Context, method string, params any) error {
	if err := t.waitEndpoint(ctx); err != nil {
		return err
	}
	t.mu.Lock()
	endpoint := t.endpoint
	t.mu.Unlock()
	return t.post(ctx, endpoint, rpcRequest{JSONRPC: "2.0", Method: method, Params: params})
}

// post sends one JSON-RPC message to the advertised endpoint.
func (t *sseTransport) post(ctx context.Context, endpoint string, msg rpcRequest) error {
	body, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}
	resp, err := t.client.Do(req)
	if err != nil {
		return err
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("http %d posting to MCP endpoint", resp.StatusCode)
	}
	return nil
}

func (t *sseTransport) close() error {
	t.cancel()
	if t.body != nil {
		_ = t.body.Close()
	}
	<-t.closed
	return nil
}
