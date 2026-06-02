// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"sync"
	"time"
)

// protocolVersion is the MCP protocol revision the client announces.
const protocolVersion = "2024-11-05"

// Tool is a tool discovered from an MCP server. InputSchema is the raw JSON
// schema the server reported (an empty object when none was given).
type Tool struct {
	ServerName  string
	Name        string
	Description string
	InputSchema json.RawMessage
}

// FullName is the namespaced "server__tool" identifier exposed to the model.
func (t Tool) FullName() string { return t.ServerName + "__" + t.Name }

// ToolResult is the outcome of a tool call. Content is the raw JSON value the
// server returned (a string, object, or array), or null.
type ToolResult struct {
	ToolName     string
	Content      json.RawMessage
	IsError      bool
	ErrorMessage string
}

// transport is a live JSON-RPC 2.0 connection to one server. call issues a
// request and returns the result payload; notify sends a notification with no
// reply; close tears the connection down.
type transport interface {
	call(ctx context.Context, method string, params any) (json.RawMessage, error)
	notify(ctx context.Context, method string, params any) error
	close() error
}

// dialer opens a transport for a server configuration.
type dialer func(ctx context.Context, cfg ServerConfig) (transport, error)

// Client connects to a single MCP server, discovers its tools, and calls them.
// It is safe for concurrent use.
type Client struct {
	config ServerConfig
	dial   dialer

	mu          sync.Mutex
	tr          transport
	tools       []Tool
	state       ServerState
	errMsg      string
	lastConnect float64
}

// NewClient builds a client for a server, using the live transport dialer.
func NewClient(cfg ServerConfig) *Client {
	return &Client{config: cfg, dial: dialReal, state: StateDisconnected}
}

// Name returns the server name.
func (c *Client) Name() string { return c.config.Name }

// State returns the current connection state.
func (c *Client) State() ServerState {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.state
}

// IsConnected reports whether the client is connected.
func (c *Client) IsConnected() bool { return c.State() == StateConnected }

// Tools returns the tools discovered on the last successful connect.
func (c *Client) Tools() []Tool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]Tool(nil), c.tools...)
}

// Status reports the server status for the API.
func (c *Client) Status() ServerStatus {
	c.mu.Lock()
	defer c.mu.Unlock()
	return ServerStatus{
		Name:        c.config.Name,
		State:       c.state,
		Transport:   c.config.Transport,
		ToolsCount:  len(c.tools),
		Error:       c.errMsg,
		LastConnect: c.lastConnect,
	}
}

// Connect opens the transport, runs the initialize handshake, and discovers the
// server's tools. A disabled server is skipped (false, no error). On failure the
// client transitions to the error state and the transport is cleaned up.
func (c *Client) Connect(ctx context.Context) (bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.state == StateConnected {
		return true, nil
	}
	if !c.config.Enabled {
		return false, nil
	}

	c.state = StateConnecting
	c.errMsg = ""

	tr, err := c.dial(ctx, c.config)
	if err != nil {
		c.fail(err)
		return false, err
	}
	c.tr = tr

	if err := c.initialize(ctx); err != nil {
		c.cleanup()
		c.fail(err)
		return false, err
	}
	c.discoverTools(ctx)

	c.state = StateConnected
	c.lastConnect = float64(time.Now().Unix())
	return true, nil
}

// initialize runs the MCP initialize request and the initialized notification.
func (c *Client) initialize(ctx context.Context) error {
	params := map[string]any{
		"protocolVersion": protocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "fastmlx", "version": "0"},
	}
	if _, err := c.tr.call(ctx, "initialize", params); err != nil {
		return err
	}
	return c.tr.notify(ctx, "notifications/initialized", map[string]any{})
}

// discoverTools lists the server's tools. A discovery failure is non-fatal (the
// server stays connected with an empty tool set), matching the reference.
func (c *Client) discoverTools(ctx context.Context) {
	raw, err := c.tr.call(ctx, "tools/list", map[string]any{})
	if err != nil {
		c.tools = nil
		return
	}
	var res struct {
		Tools []struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			InputSchema json.RawMessage `json:"inputSchema"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		c.tools = nil
		return
	}
	c.tools = c.tools[:0]
	for _, t := range res.Tools {
		c.tools = append(c.tools, Tool{
			ServerName:  c.config.Name,
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		})
	}
}

// CallTool invokes a tool by its server-local name. A disconnected client or a
// transport error yields an error result rather than a Go error, mirroring the
// reference, so the caller can surface it to the model uniformly.
func (c *Client) CallTool(ctx context.Context, toolName string, arguments json.RawMessage, timeout float64) ToolResult {
	c.mu.Lock()
	tr := c.tr
	connected := c.state == StateConnected
	c.mu.Unlock()

	if !connected || tr == nil {
		return ToolResult{
			ToolName: toolName, IsError: true,
			ErrorMessage: fmt.Sprintf("Not connected to server '%s'", c.config.Name),
		}
	}

	if timeout <= 0 {
		timeout = c.config.Timeout
	}
	callCtx, cancel := context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))
	defer cancel()

	if len(arguments) == 0 {
		arguments = json.RawMessage("{}")
	}
	params := map[string]any{"name": toolName, "arguments": arguments}
	raw, err := tr.call(callCtx, "tools/call", params)
	if err != nil {
		if callCtx.Err() == context.DeadlineExceeded {
			return ToolResult{
				ToolName: toolName, IsError: true,
				ErrorMessage: fmt.Sprintf("Tool call timed out after %gs", timeout),
			}
		}
		return ToolResult{ToolName: toolName, IsError: true, ErrorMessage: err.Error()}
	}

	content, isErr := extractContent(raw)
	return ToolResult{ToolName: toolName, Content: content, IsError: isErr}
}

// extractContent pulls the content out of a tools/call result. Text items yield
// their text, data items their data; a single item is unwrapped, multiple items
// become an array. With no content it falls back to structuredContent, then
// null. The isError flag is read from the result.
func extractContent(raw json.RawMessage) (json.RawMessage, bool) {
	var res struct {
		Content           []json.RawMessage `json:"content"`
		StructuredContent json.RawMessage   `json:"structuredContent"`
		IsError           bool              `json:"isError"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		return json.RawMessage("null"), false
	}
	if len(res.Content) == 0 {
		if len(res.StructuredContent) > 0 && string(res.StructuredContent) != "null" {
			return res.StructuredContent, res.IsError
		}
		return json.RawMessage("null"), res.IsError
	}
	items := make([]json.RawMessage, 0, len(res.Content))
	for _, item := range res.Content {
		var probe struct {
			Text *json.RawMessage `json:"text"`
			Data *json.RawMessage `json:"data"`
		}
		_ = json.Unmarshal(item, &probe)
		switch {
		case probe.Text != nil:
			items = append(items, *probe.Text)
		case probe.Data != nil:
			items = append(items, *probe.Data)
		default:
			items = append(items, item)
		}
	}
	if len(items) == 1 {
		return items[0], res.IsError
	}
	out, _ := json.Marshal(items)
	return out, res.IsError
}

// RefreshTools re-discovers tools on a connected client.
func (c *Client) RefreshTools(ctx context.Context) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.state != StateConnected {
		return
	}
	c.discoverTools(ctx)
}

// Disconnect closes the transport and clears the tool set.
func (c *Client) Disconnect() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.state == StateDisconnected {
		return
	}
	c.cleanup()
	c.state = StateDisconnected
	c.tools = nil
}

// fail records an error state. The caller holds the lock.
func (c *Client) fail(err error) {
	c.state = StateError
	c.errMsg = err.Error()
	c.cleanup()
}

// cleanup closes the transport if present. The caller holds the lock.
func (c *Client) cleanup() {
	if c.tr != nil {
		_ = c.tr.close()
		c.tr = nil
	}
}

// dialReal opens the configured transport. Stdio launches a subprocess; the SSE
// and streamable-http transports are HTTP variants of the same JSON-RPC layer.
func dialReal(ctx context.Context, cfg ServerConfig) (transport, error) {
	switch cfg.Transport {
	case TransportStdio:
		return dialStdio(ctx, cfg)
	case TransportSSE:
		return dialSSE(ctx, cfg)
	case TransportStreamableHTTP:
		return dialStreamableHTTP(ctx, cfg)
	default:
		return nil, fmt.Errorf("unknown transport: %s", cfg.Transport)
	}
}

// stdioTransport speaks newline-delimited JSON-RPC 2.0 to a subprocess over its
// stdin and stdout, the MCP stdio framing.
type stdioTransport struct {
	cmd  *exec.Cmd
	in   *json.Encoder
	out  *bufio.Reader
	mu   sync.Mutex
	next int64
}

// dialStdio launches the server subprocess and returns its transport.
func dialStdio(ctx context.Context, cfg ServerConfig) (transport, error) {
	cmd := exec.Command(cfg.Command, cfg.Args...)
	if cfg.Env != nil {
		env := os.Environ()
		for k, v := range cfg.Env {
			env = append(env, k+"="+v)
		}
		cmd.Env = env
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &stdioTransport{
		cmd: cmd,
		in:  json.NewEncoder(stdin),
		out: bufio.NewReader(stdout),
	}, nil
}

// rpcRequest is a JSON-RPC 2.0 request or notification.
type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      *int64 `json:"id,omitempty"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

// rpcResponse is a JSON-RPC 2.0 response.
type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int64          `json:"id"`
	Result  json.RawMessage `json:"result"`
	Error   *rpcError       `json:"error"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *rpcError) Error() string { return fmt.Sprintf("JSON-RPC error %d: %s", e.Code, e.Message) }

func (t *stdioTransport) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.next++
	id := t.next
	if err := t.write(rpcRequest{JSONRPC: "2.0", ID: &id, Method: method, Params: params}); err != nil {
		return nil, err
	}
	// Read until the matching response id (skipping any notifications the
	// server may interleave).
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		resp, err := t.read(ctx)
		if err != nil {
			return nil, err
		}
		if resp.ID == nil || *resp.ID != id {
			continue
		}
		if resp.Error != nil {
			return nil, resp.Error
		}
		return resp.Result, nil
	}
}

func (t *stdioTransport) notify(ctx context.Context, method string, params any) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.write(rpcRequest{JSONRPC: "2.0", Method: method, Params: params})
}

// write encodes one JSON-RPC message followed by a newline.
func (t *stdioTransport) write(msg rpcRequest) error {
	return t.in.Encode(msg) // json.Encoder appends a newline
}

// read reads one JSON-RPC message. The read runs on a goroutine so a cancelled
// context can return promptly while the subprocess pipe stays owned by the
// reader.
func (t *stdioTransport) read(ctx context.Context) (rpcResponse, error) {
	type result struct {
		resp rpcResponse
		err  error
	}
	ch := make(chan result, 1)
	go func() {
		line, err := t.out.ReadBytes('\n')
		if err != nil && len(line) == 0 {
			ch <- result{err: err}
			return
		}
		var resp rpcResponse
		if uerr := json.Unmarshal(line, &resp); uerr != nil {
			ch <- result{err: uerr}
			return
		}
		ch <- result{resp: resp}
	}()
	select {
	case <-ctx.Done():
		return rpcResponse{}, ctx.Err()
	case r := <-ch:
		return r.resp, r.err
	}
}

func (t *stdioTransport) close() error {
	if t.cmd.Process != nil {
		_ = t.cmd.Process.Kill()
	}
	return t.cmd.Wait()
}
