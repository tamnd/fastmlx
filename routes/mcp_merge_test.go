// SPDX-License-Identifier: MIT OR Apache-2.0

package routes

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tamnd/fastmlx/api"
	"github.com/tamnd/fastmlx/engine"
	"github.com/tamnd/fastmlx/mcp"
	"github.com/tamnd/fastmlx/tokenizer"
)

// recordEngine captures the tools passed to BuildPrompt and then short-circuits
// Submit, so a handler test can inspect the merged tool list without running a
// full generation.
type recordEngine struct{ gotTools []engine.Tool }

func (e *recordEngine) ModelName() string              { return "rec" }
func (e *recordEngine) Tokenizer() tokenizer.Tokenizer { return nil }
func (e *recordEngine) CountTokens(string) int         { return 0 }
func (e *recordEngine) BuildPrompt(_ []engine.Message, tools []engine.Tool, _ engine.PromptOptions) (string, error) {
	e.gotTools = tools
	return "prompt", nil
}
func (e *recordEngine) Submit(*engine.Request) (<-chan engine.RequestOutput, error) {
	return nil, engine.ErrQueueFull // stop after BuildPrompt; the handler writes 503
}
func (e *recordEngine) Abort(string)                    {}
func (e *recordEngine) Defaults() engine.SamplingParams { return engine.SamplingParams{} }
func (e *recordEngine) InFlight() int                   { return 0 }

func userTool(name, desc, params string) api.Tool {
	t := api.Tool{Type: "function"}
	t.Function.Name = name
	t.Function.Description = desc
	if params != "" {
		t.Function.Parameters = json.RawMessage(params)
	}
	return t
}

func TestMCPToolSchemaFallback(t *testing.T) {
	cases := []struct {
		name string
		in   json.RawMessage
		want string
	}{
		{"absent", nil, `{"type":"object","properties":{}}`},
		{"empty object", json.RawMessage(`{}`), `{"type":"object","properties":{}}`},
		{"null", json.RawMessage(`null`), `{"type":"object","properties":{}}`},
		{"populated", json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`),
			`{"type":"object","properties":{"x":{"type":"string"}}}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := string(mcpToolSchema(c.in)); got != c.want {
				t.Errorf("mcpToolSchema(%s) = %s, want %s", c.in, got, c.want)
			}
		})
	}
}

func TestMergeMCPToolsOrderAndNamespacing(t *testing.T) {
	mcpTools := []mcp.Tool{
		{ServerName: "fs", Name: "read", Description: "read a file", InputSchema: json.RawMessage(`{"type":"object","properties":{"p":{}}}`)},
		{ServerName: "db", Name: "query"},
	}
	user := []api.Tool{userTool("get_weather", "weather", `{"type":"object"}`)}

	got := mergeMCPTools(mcpTools, user)
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	// MCP tools first in discovery order, namespaced; user tool appended.
	if got[0].Name != "fs__read" || got[1].Name != "db__query" || got[2].Name != "get_weather" {
		t.Fatalf("names = %q, %q, %q", got[0].Name, got[1].Name, got[2].Name)
	}
	// db__query had no schema, so it gets the empty-object fallback.
	if string(got[1].Parameters.(json.RawMessage)) != `{"type":"object","properties":{}}` {
		t.Errorf("db__query params = %s", got[1].Parameters)
	}
}

func TestMergeMCPToolsUserOverridesInPlace(t *testing.T) {
	// A user tool named exactly like an MCP tool overrides it but keeps its slot.
	mcpTools := []mcp.Tool{
		{ServerName: "fs", Name: "read", Description: "mcp version"},
		{ServerName: "db", Name: "query", Description: "db version"},
	}
	user := []api.Tool{userTool("fs__read", "user override", `{"type":"object"}`)}

	got := mergeMCPTools(mcpTools, user)
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2 (override should not append)", len(got))
	}
	if got[0].Name != "fs__read" || got[0].Description != "user override" {
		t.Errorf("slot 0 = %+v, want the user override in place", got[0])
	}
	if got[1].Name != "db__query" {
		t.Errorf("slot 1 = %q", got[1].Name)
	}
}

func TestMergeMCPToolsSkipsUnnamedUserTool(t *testing.T) {
	got := mergeMCPTools(nil, []api.Tool{userTool("", "no name", ""), userTool("ok", "kept", "")})
	if len(got) != 1 || got[0].Name != "ok" {
		t.Fatalf("got = %+v, want only the named tool", got)
	}
}

// TestChatCompletionsInjectsMCPTools checks the wiring: with a manager attached,
// the chat handler feeds the merged MCP+user tools into BuildPrompt.
func TestChatCompletionsInjectsMCPTools(t *testing.T) {
	eng := &recordEngine{}
	rt := New(eng)
	rt.SetMCPManager(&fakeMCP{tools: []mcp.Tool{
		{ServerName: "fs", Name: "read", Description: "read a file"},
	}})

	body := `{"model":"rec","messages":[{"role":"user","content":"hi"}],` +
		`"tools":[{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object"}}}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rt.ChatCompletions(httptest.NewRecorder(), r)

	if len(eng.gotTools) != 2 {
		t.Fatalf("BuildPrompt saw %d tools, want 2 (1 MCP + 1 user)", len(eng.gotTools))
	}
	if eng.gotTools[0].Name != "fs__read" || eng.gotTools[1].Name != "get_weather" {
		t.Errorf("tool names = %q, %q", eng.gotTools[0].Name, eng.gotTools[1].Name)
	}
}

// TestChatCompletionsNoManagerKeepsUserTools confirms the unmerged path: with no
// manager, only the user-supplied tools reach BuildPrompt.
func TestChatCompletionsNoManagerKeepsUserTools(t *testing.T) {
	eng := &recordEngine{}
	rt := New(eng)

	body := `{"model":"rec","messages":[{"role":"user","content":"hi"}],` +
		`"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rt.ChatCompletions(httptest.NewRecorder(), r)

	if len(eng.gotTools) != 1 || eng.gotTools[0].Name != "get_weather" {
		t.Fatalf("gotTools = %+v, want only get_weather", eng.gotTools)
	}
}
