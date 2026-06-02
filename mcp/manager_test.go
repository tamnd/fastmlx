// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"testing"
)

// newManagerWithFakes builds a manager whose clients all dial the given
// per-server fake transports, keyed by server name.
func newManagerWithFakes(cfg Config, fakes map[string]*fakeTransport) *Manager {
	m := NewManager(cfg)
	for _, c := range m.clients {
		name := c.config.Name
		f := fakes[name]
		if f == nil {
			f = &fakeTransport{}
		}
		c.dial = func(ctx context.Context, _ ServerConfig) (transport, error) { return f, nil }
	}
	return m
}

func twoServerConfig() Config {
	return Config{
		DefaultTimeout: 5,
		MaxToolCalls:   10,
		Servers: []ServerConfig{
			{Name: "fs", Transport: TransportStdio, Command: "x", Enabled: true, Timeout: 5},
			{Name: "db", Transport: TransportStdio, Command: "y", Enabled: true, Timeout: 5},
		},
	}
}

func TestManagerStartAggregatesTools(t *testing.T) {
	fakes := map[string]*fakeTransport{
		"fs": {tools: `{"tools":[{"name":"read","description":"r","inputSchema":{}}]}`,
			callResult: `{"content":[{"type":"text","text":"file data"}]}`},
		"db": {tools: `{"tools":[{"name":"query","description":"q","inputSchema":{}}]}`,
			callResult: `{"content":[{"type":"text","text":"rows"}]}`},
	}
	m := newManagerWithFakes(twoServerConfig(), fakes)
	m.Start(context.Background())
	defer m.Stop()

	if !m.IsStarted() {
		t.Fatal("not started")
	}
	tools := m.AllTools()
	if len(tools) != 2 {
		t.Fatalf("tools = %d", len(tools))
	}
	if tools[0].FullName() != "fs__read" || tools[1].FullName() != "db__query" {
		t.Errorf("aggregated = %q,%q", tools[0].FullName(), tools[1].FullName())
	}
	stats := m.ServerStatuses()
	if len(stats) != 2 || stats[0].State != StateConnected || stats[1].State != StateConnected {
		t.Errorf("statuses = %+v", stats)
	}
}

func TestManagerExecuteToolByFullName(t *testing.T) {
	fakes := map[string]*fakeTransport{
		"fs": {tools: `{"tools":[{"name":"read","inputSchema":{}}]}`,
			callResult: `{"content":[{"type":"text","text":"file data"}]}`},
		"db": {tools: `{"tools":[{"name":"query","inputSchema":{}}]}`,
			callResult: `{"content":[{"type":"text","text":"rows"}]}`},
	}
	m := newManagerWithFakes(twoServerConfig(), fakes)
	m.Start(context.Background())
	defer m.Stop()

	res := m.ExecuteTool(context.Background(), "db__query", json.RawMessage(`{"sql":"select 1"}`), 0)
	if res.IsError || string(res.Content) != `"rows"` {
		t.Fatalf("result = %+v", res)
	}
	if fakes["db"].lastName != "query" {
		t.Errorf("routed to wrong tool: %q", fakes["db"].lastName)
	}
}

func TestManagerExecuteToolBareNameLookup(t *testing.T) {
	fakes := map[string]*fakeTransport{
		"fs": {tools: `{"tools":[{"name":"read","inputSchema":{}}]}`,
			callResult: `{"content":[{"type":"text","text":"ok"}]}`},
		"db": {tools: `{"tools":[{"name":"query","inputSchema":{}}]}`},
	}
	m := newManagerWithFakes(twoServerConfig(), fakes)
	m.Start(context.Background())
	defer m.Stop()

	// "read" has no server prefix; it is found on fs.
	res := m.ExecuteTool(context.Background(), "read", nil, 0)
	if res.IsError || string(res.Content) != `"ok"` {
		t.Fatalf("result = %+v", res)
	}
}

func TestManagerExecuteToolUnknown(t *testing.T) {
	m := newManagerWithFakes(twoServerConfig(), nil)
	m.Start(context.Background())
	defer m.Stop()

	res := m.ExecuteTool(context.Background(), "nope", nil, 0)
	if !res.IsError || res.ErrorMessage != "Tool 'nope' not found in any connected server" {
		t.Errorf("result = %+v", res)
	}
}

func TestManagerExecuteToolUnknownServer(t *testing.T) {
	m := newManagerWithFakes(twoServerConfig(), nil)
	m.Start(context.Background())
	defer m.Stop()

	res := m.ExecuteTool(context.Background(), "ghost__x", nil, 0)
	if !res.IsError || res.ErrorMessage != "Server 'ghost' not found" {
		t.Errorf("result = %+v", res)
	}
}

func TestManagerHasTool(t *testing.T) {
	fakes := map[string]*fakeTransport{
		"fs": {tools: `{"tools":[{"name":"read","inputSchema":{}}]}`},
	}
	m := newManagerWithFakes(twoServerConfig(), fakes)
	m.Start(context.Background())
	defer m.Stop()

	if !m.HasTool("fs__read") || !m.HasTool("read") {
		t.Error("expected fs__read and bare read to be found")
	}
	if m.HasTool("missing") {
		t.Error("missing tool should not be found")
	}
}
