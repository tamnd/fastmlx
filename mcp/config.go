// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ConfigEnvVar is the environment variable that points at an MCP config file.
const ConfigEnvVar = "FASTMLX_MCP_CONFIG"

// configSearchPaths are the default locations searched when no path is given.
// These are fastmlx-native; a Claude Desktop config (the "mcpServers" shape) is
// still accepted from any of them.
var configSearchPaths = []string{
	"./mcp.json",
	"~/.config/fastmlx/mcp.json",
}

// allowedServerFields is the set of keys a server entry may carry; any other key
// is a configuration error (the reference surfaces these as a TypeError on the
// dataclass constructor).
var allowedServerFields = map[string]bool{
	"name": true, "transport": true, "command": true, "args": true,
	"env": true, "url": true, "headers": true, "enabled": true, "timeout": true,
}

// LoadConfig loads MCP configuration from a file. The search order is the
// explicit path, then the FASTMLX_MCP_CONFIG environment variable, then the
// default search paths. When nothing is found it returns an empty config (no
// servers) rather than an error.
func LoadConfig(path string) (Config, error) {
	configPath, err := findConfigFile(path)
	if err != nil {
		return Config{}, err
	}
	if configPath == "" {
		return Config{MaxToolCalls: 10, DefaultTimeout: 30.0}, nil
	}
	content, err := os.ReadFile(configPath)
	if err != nil {
		return Config{}, err
	}
	return ValidateConfig(content)
}

// findConfigFile resolves which config file to use, returning "" when none of
// the default locations exist. An explicit path that does not exist is an error;
// a missing environment-variable path is a warning (skipped), matching the
// reference.
func findConfigFile(explicit string) (string, error) {
	if explicit != "" {
		p := expandUser(explicit)
		if fileExists(p) {
			return p, nil
		}
		return "", fmt.Errorf("MCP config file not found: %s", explicit)
	}
	if env := os.Getenv(ConfigEnvVar); env != "" {
		p := expandUser(env)
		if fileExists(p) {
			return p, nil
		}
	}
	for _, sp := range configSearchPaths {
		p := expandUser(sp)
		if fileExists(p) {
			return p, nil
		}
	}
	return "", nil
}

// ValidateConfig parses and validates a raw JSON config document. It accepts
// both the fastmlx "servers" shape and the Claude Desktop "mcpServers" shape,
// preserving server definition order.
func ValidateConfig(data []byte) (Config, error) {
	if !isJSONObject(data) {
		return Config{}, fmt.Errorf("MCP config must be a dictionary")
	}

	_, fields, err := orderedObject(data)
	if err != nil {
		return Config{}, err
	}

	serversRaw := fields["servers"]
	if isEmptyObject(serversRaw) || len(serversRaw) == 0 {
		// "servers" absent or empty falls through to "mcpServers".
		if mcp, ok := fields["mcpServers"]; ok {
			serversRaw = mcp
		}
	}

	var servers []ServerConfig
	if len(serversRaw) > 0 && string(serversRaw) != "null" {
		if !isJSONObject(serversRaw) {
			return Config{}, fmt.Errorf("'servers' must be a dictionary")
		}
		names, entries, err := orderedObject(serversRaw)
		if err != nil {
			return Config{}, err
		}
		for _, name := range names {
			sc, err := parseServer(name, entries[name])
			if err != nil {
				return Config{}, err
			}
			servers = append(servers, sc)
		}
	}

	maxToolCalls := 10
	if raw, ok := fields["max_tool_calls"]; ok {
		var n json.Number
		if err := json.Unmarshal(raw, &n); err != nil {
			return Config{}, fmt.Errorf("'max_tool_calls' must be a positive integer")
		}
		i, err := n.Int64()
		if err != nil || i < 1 {
			return Config{}, fmt.Errorf("'max_tool_calls' must be a positive integer")
		}
		maxToolCalls = int(i)
	}

	defaultTimeout := 30.0
	if raw, ok := fields["default_timeout"]; ok {
		var f float64
		if err := json.Unmarshal(raw, &f); err != nil || f <= 0 {
			return Config{}, fmt.Errorf("'default_timeout' must be a positive number")
		}
		defaultTimeout = f
	}

	return Config{Servers: servers, MaxToolCalls: maxToolCalls, DefaultTimeout: defaultTimeout}, nil
}

// parseServer builds and validates one server config from its raw entry. The
// name comes from the map key. Unknown fields, an unknown transport, or a
// missing required field are reported as configuration errors.
func parseServer(name string, raw json.RawMessage) (ServerConfig, error) {
	if !isJSONObject(raw) {
		return ServerConfig{}, fmt.Errorf("Server '%s' config must be a dictionary", name)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return ServerConfig{}, fmt.Errorf("invalid config for server '%s': %v", name, err)
	}
	for k := range fields {
		if !allowedServerFields[k] {
			return ServerConfig{}, fmt.Errorf("invalid config for server '%s': unexpected field '%s'", name, k)
		}
	}

	var entry struct {
		Transport *string           `json:"transport"`
		Command   *string           `json:"command"`
		Args      []string          `json:"args"`
		Env       map[string]string `json:"env"`
		URL       *string           `json:"url"`
		Headers   map[string]string `json:"headers"`
		Enabled   *bool             `json:"enabled"`
		Timeout   *float64          `json:"timeout"`
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	if err := dec.Decode(&entry); err != nil {
		return ServerConfig{}, fmt.Errorf("invalid config for server '%s': %v", name, err)
	}

	sc := ServerConfig{Name: name, Transport: TransportStdio, Enabled: true, Timeout: 30.0}
	if entry.Transport != nil {
		t, err := ParseTransport(*entry.Transport)
		if err != nil {
			return ServerConfig{}, err
		}
		sc.Transport = t
	}
	if entry.Command != nil {
		sc.Command = *entry.Command
	}
	sc.Args = entry.Args
	sc.Env = entry.Env
	if entry.URL != nil {
		sc.URL = *entry.URL
	}
	sc.Headers = entry.Headers
	if entry.Enabled != nil {
		sc.Enabled = *entry.Enabled
	}
	if entry.Timeout != nil {
		sc.Timeout = *entry.Timeout
	}
	if err := sc.Validate(); err != nil {
		return ServerConfig{}, err
	}
	return sc, nil
}

// orderedObject returns the keys of a JSON object in source order alongside a
// map of their raw values.
func orderedObject(data []byte) ([]string, map[string]json.RawMessage, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	if _, err := dec.Token(); err != nil { // opening {
		return nil, nil, err
	}
	var keys []string
	values := map[string]json.RawMessage{}
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil, nil, err
		}
		key := kt.(string)
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return nil, nil, err
		}
		keys = append(keys, key)
		values[key] = raw
	}
	if _, err := dec.Token(); err != nil { // closing }
		return nil, nil, err
	}
	return keys, values, nil
}

func isJSONObject(data []byte) bool {
	t := bytes.TrimSpace(data)
	return len(t) > 0 && t[0] == '{'
}

func isEmptyObject(data []byte) bool {
	return string(bytes.TrimSpace(data)) == "{}"
}

func fileExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}

// expandUser resolves a leading ~ to the user's home directory.
func expandUser(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			if p == "~" {
				return home
			}
			return filepath.Join(home, p[2:])
		}
	}
	return p
}

// CreateExampleConfig returns an example MCP configuration as indented JSON, the
// same shape the reference emits (two-space indent).
func CreateExampleConfig() string {
	return exampleConfig
}

// exampleConfig is the example document, formatted to match json.dumps(indent=2).
const exampleConfig = `{
  "servers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/tmp"
      ],
      "enabled": true,
      "timeout": 30
    },
    "web-search": {
      "transport": "sse",
      "url": "http://localhost:3001/sse",
      "enabled": true,
      "timeout": 60
    },
    "sqlite": {
      "transport": "stdio",
      "command": "uvx",
      "args": [
        "mcp-server-sqlite",
        "--db-path",
        "data.db"
      ],
      "enabled": true
    }
  },
  "max_tool_calls": 10,
  "default_timeout": 30.0
}`
