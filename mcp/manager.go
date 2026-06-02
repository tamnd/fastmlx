// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

// Manager owns the clients for every configured server and presents a single
// view over them: aggregated tools, per-server status, and tool execution
// routed to the owning server.
type Manager struct {
	config  Config
	clients []*Client // definition order

	mu      sync.Mutex
	started bool
}

// NewManager builds a manager and a client per configured server.
func NewManager(config Config) *Manager {
	m := &Manager{config: config}
	for _, sc := range config.Servers {
		m.clients = append(m.clients, NewClient(sc))
	}
	return m
}

// IsStarted reports whether Start has run.
func (m *Manager) IsStarted() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.started
}

// Start connects to every enabled server in parallel. It is idempotent.
func (m *Manager) Start(ctx context.Context) {
	m.mu.Lock()
	if m.started {
		m.mu.Unlock()
		return
	}
	m.mu.Unlock()

	var wg sync.WaitGroup
	for _, c := range m.clients {
		if !c.config.Enabled {
			continue
		}
		wg.Add(1)
		go func(c *Client) {
			defer wg.Done()
			_, _ = c.Connect(ctx)
		}(c)
	}
	wg.Wait()

	m.mu.Lock()
	m.started = true
	m.mu.Unlock()
}

// Stop disconnects from every server in parallel. It is idempotent.
func (m *Manager) Stop() {
	m.mu.Lock()
	if !m.started {
		m.mu.Unlock()
		return
	}
	m.mu.Unlock()

	var wg sync.WaitGroup
	for _, c := range m.clients {
		wg.Add(1)
		go func(c *Client) {
			defer wg.Done()
			c.Disconnect()
		}(c)
	}
	wg.Wait()

	m.mu.Lock()
	m.started = false
	m.mu.Unlock()
}

// AllTools returns the tools from every connected server, in server order.
func (m *Manager) AllTools() []Tool {
	var tools []Tool
	for _, c := range m.clients {
		if c.IsConnected() {
			tools = append(tools, c.Tools()...)
		}
	}
	return tools
}

// ServerStatuses returns the status of every server, in definition order.
func (m *Manager) ServerStatuses() []ServerStatus {
	out := make([]ServerStatus, 0, len(m.clients))
	for _, c := range m.clients {
		out = append(out, c.Status())
	}
	return out
}

// client returns the client for a server name, or nil.
func (m *Manager) client(name string) *Client {
	for _, c := range m.clients {
		if c.config.Name == name {
			return c
		}
	}
	return nil
}

// ExecuteTool runs a tool by its full name (server__tool). When the name has no
// server prefix the tool is looked up across connected servers. Unknown tools,
// unknown servers, and disconnected servers yield an error result.
func (m *Manager) ExecuteTool(ctx context.Context, fullName string, arguments json.RawMessage, timeout float64) ToolResult {
	serverName, toolName := splitToolName(fullName)
	if serverName == "" {
		serverName = m.findToolServer(fullName)
		toolName = fullName
	}
	if serverName == "" {
		return ToolResult{
			ToolName: fullName, IsError: true,
			ErrorMessage: fmt.Sprintf("Tool '%s' not found in any connected server", fullName),
		}
	}
	c := m.client(serverName)
	if c == nil {
		return ToolResult{
			ToolName: fullName, IsError: true,
			ErrorMessage: fmt.Sprintf("Server '%s' not found", serverName),
		}
	}
	if !c.IsConnected() {
		return ToolResult{
			ToolName: fullName, IsError: true,
			ErrorMessage: fmt.Sprintf("Server '%s' is not connected", serverName),
		}
	}
	if timeout <= 0 {
		timeout = m.config.DefaultTimeout
	}
	return c.CallTool(ctx, toolName, arguments, timeout)
}

// splitToolName splits a "server__tool" name on the first "__". A name with no
// separator has an empty server.
func splitToolName(fullName string) (server, tool string) {
	if before, after, found := strings.Cut(fullName, "__"); found {
		return before, after
	}
	return "", fullName
}

// findToolServer returns the name of the first connected server exposing a tool
// with the given local name, or "".
func (m *Manager) findToolServer(toolName string) string {
	for _, c := range m.clients {
		if !c.IsConnected() {
			continue
		}
		for _, t := range c.Tools() {
			if t.Name == toolName {
				return c.config.Name
			}
		}
	}
	return ""
}

// HasTool reports whether any connected server exposes the tool, by full name or
// by bare tool name when the name is un-namespaced.
func (m *Manager) HasTool(fullName string) bool {
	for _, t := range m.AllTools() {
		if t.FullName() == fullName {
			return true
		}
	}
	if !strings.Contains(fullName, "__") {
		for _, t := range m.AllTools() {
			if t.Name == fullName {
				return true
			}
		}
	}
	return false
}
