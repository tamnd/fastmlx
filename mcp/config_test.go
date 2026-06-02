// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"os"
	"path/filepath"
	"testing"
)

func TestValidateConfigServersShape(t *testing.T) {
	data := []byte(`{
		"servers": {
			"fs": {"transport": "stdio", "command": "npx", "args": ["-y", "srv"]},
			"search": {"transport": "sse", "url": "http://localhost:3001/sse", "timeout": 60}
		},
		"max_tool_calls": 5,
		"default_timeout": 15
	}`)
	cfg, err := ValidateConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MaxToolCalls != 5 || cfg.DefaultTimeout != 15 {
		t.Fatalf("scalars = %d/%v", cfg.MaxToolCalls, cfg.DefaultTimeout)
	}
	if len(cfg.Servers) != 2 {
		t.Fatalf("servers = %d", len(cfg.Servers))
	}
	// Definition order is preserved.
	if cfg.Servers[0].Name != "fs" || cfg.Servers[1].Name != "search" {
		t.Errorf("order = %q,%q", cfg.Servers[0].Name, cfg.Servers[1].Name)
	}
	fs := cfg.Servers[0]
	if fs.Transport != TransportStdio || fs.Command != "npx" || len(fs.Args) != 2 {
		t.Errorf("fs = %+v", fs)
	}
	// Defaults applied where omitted.
	if !fs.Enabled || fs.Timeout != 30.0 {
		t.Errorf("fs defaults: enabled=%v timeout=%v", fs.Enabled, fs.Timeout)
	}
	if cfg.Servers[1].Timeout != 60 {
		t.Errorf("search timeout = %v", cfg.Servers[1].Timeout)
	}
}

func TestValidateConfigClaudeDesktopShape(t *testing.T) {
	// "mcpServers" is accepted when "servers" is absent.
	data := []byte(`{"mcpServers": {"db": {"command": "uvx", "args": ["mcp-server-sqlite"]}}}`)
	cfg, err := ValidateConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Servers) != 1 || cfg.Servers[0].Name != "db" {
		t.Fatalf("servers = %+v", cfg.Servers)
	}
	if cfg.MaxToolCalls != 10 || cfg.DefaultTimeout != 30.0 {
		t.Errorf("defaults = %d/%v", cfg.MaxToolCalls, cfg.DefaultTimeout)
	}
}

func TestValidateConfigEmptyServersFallsThrough(t *testing.T) {
	data := []byte(`{"servers": {}, "mcpServers": {"x": {"command": "echo"}}}`)
	cfg, err := ValidateConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Servers) != 1 || cfg.Servers[0].Name != "x" {
		t.Fatalf("servers = %+v", cfg.Servers)
	}
}

func TestValidateConfigErrors(t *testing.T) {
	cases := []struct {
		name string
		data string
		want string
	}{
		{"not_object", `[]`, "MCP config must be a dictionary"},
		{"servers_not_object", `{"servers": []}`, "'servers' must be a dictionary"},
		{"stdio_no_command", `{"servers": {"a": {"transport": "stdio"}}}`, "stdio transport requires 'command'"},
		{"sse_no_url", `{"servers": {"a": {"transport": "sse"}}}`, "sse transport requires 'url'"},
		{"http_no_url", `{"servers": {"a": {"transport": "streamable-http"}}}`, "streamable-http transport requires 'url'"},
		{"bad_transport", `{"servers": {"a": {"transport": "carrier-pigeon", "command": "x"}}}`, "not a valid MCPTransport"},
		{"unknown_field", `{"servers": {"a": {"command": "x", "bogus": 1}}}`, "unexpected field 'bogus'"},
		{"bad_max_tool_calls", `{"max_tool_calls": 0}`, "'max_tool_calls' must be a positive integer"},
		{"bad_max_tool_calls_type", `{"max_tool_calls": "x"}`, "'max_tool_calls' must be a positive integer"},
		{"bad_timeout", `{"default_timeout": -1}`, "'default_timeout' must be a positive number"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := ValidateConfig([]byte(c.data))
			if err == nil {
				t.Fatalf("expected error containing %q", c.want)
			}
			if !contains(err.Error(), c.want) {
				t.Errorf("error = %q, want contains %q", err.Error(), c.want)
			}
		})
	}
}

func TestLoadConfigMissingReturnsEmpty(t *testing.T) {
	// No file, no env, no default paths in this temp HOME.
	dir := t.TempDir()
	t.Setenv("HOME", dir)
	t.Setenv(ConfigEnvVar, "")
	// Run from a directory with no ./mcp.json.
	cwd, _ := os.Getwd()
	defer os.Chdir(cwd)
	os.Chdir(dir)

	cfg, err := LoadConfig("")
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Servers) != 0 || cfg.MaxToolCalls != 10 || cfg.DefaultTimeout != 30.0 {
		t.Errorf("empty config = %+v", cfg)
	}
}

func TestLoadConfigExplicitMissing(t *testing.T) {
	_, err := LoadConfig("/no/such/mcp.json")
	if err == nil || !contains(err.Error(), "MCP config file not found") {
		t.Fatalf("err = %v", err)
	}
}

func TestLoadConfigFromEnv(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "mcp.json")
	if err := os.WriteFile(p, []byte(`{"servers": {"a": {"command": "echo"}}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv(ConfigEnvVar, p)
	cfg, err := LoadConfig("")
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Servers) != 1 || cfg.Servers[0].Name != "a" {
		t.Errorf("config = %+v", cfg)
	}
}

func TestCreateExampleConfigRoundTrips(t *testing.T) {
	cfg, err := ValidateConfig([]byte(CreateExampleConfig()))
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Servers) != 3 {
		t.Fatalf("servers = %d", len(cfg.Servers))
	}
	names := []string{cfg.Servers[0].Name, cfg.Servers[1].Name, cfg.Servers[2].Name}
	want := []string{"filesystem", "web-search", "sqlite"}
	for i := range want {
		if names[i] != want[i] {
			t.Errorf("[%d] = %q, want %q", i, names[i], want[i])
		}
	}
}

func TestParseTransport(t *testing.T) {
	for _, s := range []string{"stdio", "sse", "streamable-http"} {
		if _, err := ParseTransport(s); err != nil {
			t.Errorf("ParseTransport(%q) = %v", s, err)
		}
	}
	if _, err := ParseTransport("nope"); err == nil {
		t.Error("expected error for unknown transport")
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func BenchmarkValidateConfig(b *testing.B) {
	data := []byte(CreateExampleConfig())
	b.ReportAllocs()
	for b.Loop() {
		if _, err := ValidateConfig(data); err != nil {
			b.Fatal(err)
		}
	}
}
