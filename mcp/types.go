// SPDX-License-Identifier: MIT OR Apache-2.0

// Package mcp implements the Model Context Protocol client subsystem: server
// configuration, connection lifecycle, tool discovery and aggregation across
// servers, and tool-call execution. The transport that speaks JSON-RPC to a
// server (stdio subprocess, SSE, streamable-http) is the seam in client.go; the
// config, manager, and executor around it are pure orchestration.
package mcp

import "fmt"

// Transport is an MCP server transport kind.
type Transport string

const (
	TransportStdio          Transport = "stdio"
	TransportSSE            Transport = "sse"
	TransportStreamableHTTP Transport = "streamable-http"
)

// ParseTransport resolves a transport string, rejecting unknown kinds.
func ParseTransport(s string) (Transport, error) {
	switch Transport(s) {
	case TransportStdio, TransportSSE, TransportStreamableHTTP:
		return Transport(s), nil
	default:
		return "", fmt.Errorf("'%s' is not a valid MCPTransport", s)
	}
}

// ServerState is an MCP server connection state.
type ServerState string

const (
	StateDisconnected ServerState = "disconnected"
	StateConnecting   ServerState = "connecting"
	StateConnected    ServerState = "connected"
	StateError        ServerState = "error"
)

// ServerConfig configures a single MCP server. The transport selects which
// fields are required: stdio needs a command, sse and streamable-http need a
// url. The zero value is not valid; build one through the config loader or set
// the fields and call Validate.
type ServerConfig struct {
	Name      string
	Transport Transport

	// stdio transport
	Command string
	Args    []string
	Env     map[string]string

	// sse and streamable-http transports
	URL string

	// streamable-http transport
	Headers map[string]string

	// common
	Enabled bool
	Timeout float64
}

// Validate checks the required fields for the configured transport, mirroring
// the reference's __post_init__.
func (c *ServerConfig) Validate() error {
	switch c.Transport {
	case TransportStdio:
		if c.Command == "" {
			return fmt.Errorf("MCP server '%s': stdio transport requires 'command'", c.Name)
		}
	case TransportSSE:
		if c.URL == "" {
			return fmt.Errorf("MCP server '%s': sse transport requires 'url'", c.Name)
		}
	case TransportStreamableHTTP:
		if c.URL == "" {
			return fmt.Errorf("MCP server '%s': streamable-http transport requires 'url'", c.Name)
		}
	}
	return nil
}

// Config is the root MCP client configuration. Servers keeps definition order so
// connection, status, and tool aggregation are deterministic (the reference
// relies on Python dict insertion order for the same).
type Config struct {
	Servers        []ServerConfig
	MaxToolCalls   int
	DefaultTimeout float64
}

// Server returns the configured server with the given name, or false.
func (c Config) Server(name string) (ServerConfig, bool) {
	for _, s := range c.Servers {
		if s.Name == name {
			return s, true
		}
	}
	return ServerConfig{}, false
}

// ServerStatus reports a server connection's state for the API.
type ServerStatus struct {
	Name        string
	State       ServerState
	Transport   Transport
	ToolsCount  int
	Error       string
	LastConnect float64
}
