// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"bytes"
	"encoding/json"
	"strings"
)

// This file ports the pure MCP<->OpenAI tool-schema conversions: turning a
// discovered MCP tool into an OpenAI function definition, parsing a model's
// OpenAI tool call back into a (server, tool, arguments) triple, merging MCP and
// user tools with user precedence, and pulling tool calls out of a model
// response. They are functions of their JSON inputs alone, so they carry no
// transport coupling.

// ToolToOpenAI converts a discovered MCP tool into an OpenAI function-calling
// tool definition. A missing or empty input schema falls back to the empty
// object schema, mirroring the reference's `input_schema or {default}`.
func ToolToOpenAI(t Tool) map[string]any {
	var params any = map[string]any{"type": "object", "properties": map[string]any{}}
	var m map[string]any
	if len(t.InputSchema) > 0 && json.Unmarshal(t.InputSchema, &m) == nil && len(m) > 0 {
		params = m
	}
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        t.FullName(),
			"description": t.Description,
			"parameters":  params,
		},
	}
}

// ToolsToOpenAI converts a list of MCP tools to OpenAI definitions.
func ToolsToOpenAI(tools []Tool) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		out = append(out, ToolToOpenAI(t))
	}
	return out
}

// OpenAICallToMCP parses an OpenAI tool call from a model response back into the
// server name, tool name, and arguments. The arguments come from the call's
// function.arguments: a string is JSON-parsed (an empty object on a parse
// error), and a non-string value is used as-is when truthy and an empty object
// otherwise. The namespaced name splits on the first "__"; an un-namespaced name
// leaves the server empty for the caller to resolve.
func OpenAICallToMCP(toolCall map[string]any) (serverName, toolName string, arguments any) {
	function, _ := toolCall["function"].(map[string]any)
	fullName, _ := function["name"].(string)

	argsRaw, present := function["arguments"]
	if !present {
		argsRaw = "{}"
	}
	switch a := argsRaw.(type) {
	case string:
		var parsed any
		if json.Unmarshal([]byte(a), &parsed) == nil {
			arguments = parsed
		} else {
			arguments = map[string]any{}
		}
	default:
		if truthy(argsRaw) {
			arguments = argsRaw
		} else {
			arguments = map[string]any{}
		}
	}

	if before, after, found := strings.Cut(fullName, "__"); found {
		serverName, toolName = before, after
	} else {
		serverName, toolName = "", fullName
	}
	return serverName, toolName, arguments
}

// MergeTools combines MCP tools with user-provided OpenAI tools, user tools
// taking precedence on a name conflict. MCP tools are emitted in discovery
// order; a user tool with a conflicting name replaces the value in place, and a
// new-named user tool is appended. A user tool with no function name is skipped.
func MergeTools(mcpTools []Tool, userTools []map[string]any) []map[string]any {
	order := []string{}
	byName := map[string]map[string]any{}
	for _, t := range mcpTools {
		name := t.FullName()
		if _, seen := byName[name]; !seen {
			order = append(order, name)
		}
		byName[name] = ToolToOpenAI(t)
	}
	for _, tool := range userTools {
		fn, _ := tool["function"].(map[string]any)
		name, _ := fn["name"].(string)
		if name == "" {
			continue
		}
		if _, seen := byName[name]; !seen {
			order = append(order, name)
		}
		byName[name] = tool
	}
	out := make([]map[string]any, 0, len(order))
	for _, name := range order {
		out = append(out, byName[name])
	}
	return out
}

// ExtractToolCalls returns the tool calls from an OpenAI-format model response:
// the tool_calls of the first choice's message, or an empty list when there are
// no choices or none are present.
func ExtractToolCalls(response map[string]any) []any {
	choices, _ := response["choices"].([]any)
	if len(choices) == 0 {
		return []any{}
	}
	first, _ := choices[0].(map[string]any)
	message, _ := first["message"].(map[string]any)
	calls, _ := message["tool_calls"].([]any)
	if calls == nil {
		return []any{}
	}
	return calls
}

// HasToolCalls reports whether a response carries any tool calls.
func HasToolCalls(response map[string]any) bool {
	return len(ExtractToolCalls(response)) > 0
}

// ToMessage converts a tool result into an OpenAI tool-result message. An error
// result renders as "Error: " followed by the error message; a string content is
// used verbatim; any other content is re-serialized with Python json.dumps
// formatting. The content's JSON form is the raw bytes the server returned, so a
// bare JSON string is unwrapped to its decoded text rather than re-quoted.
func (r ToolResult) ToMessage(toolCallID string) map[string]any {
	var content string
	switch {
	case r.IsError:
		content = "Error: " + r.ErrorMessage
	case isJSONString(r.Content):
		var s string
		_ = json.Unmarshal(r.Content, &s)
		content = s
	default:
		content = pyJSONDumps(r.Content)
	}
	return map[string]any{
		"role":         "tool",
		"tool_call_id": toolCallID,
		"content":      content,
	}
}

// FormatToolResult formats a single tool result as a conversation message.
func FormatToolResult(result ToolResult, toolCallID string) map[string]any {
	return result.ToMessage(toolCallID)
}

// ResultWithCallID pairs a tool result with the id of the call it answers.
type ResultWithCallID struct {
	Result     ToolResult
	ToolCallID string
}

// FormatToolResults formats several tool results as messages, in order.
func FormatToolResults(results []ResultWithCallID) []map[string]any {
	out := make([]map[string]any, 0, len(results))
	for _, rc := range results {
		out = append(out, FormatToolResult(rc.Result, rc.ToolCallID))
	}
	return out
}

// isJSONString reports whether a raw JSON value is a string literal, the case
// the reference detects with isinstance(content, str) to avoid re-quoting it.
func isJSONString(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && trimmed[0] == '"'
}

// truthy mirrors Python truthiness for a JSON-decoded value, used for the
// non-string arguments fallback (`arguments_str or {}`).
func truthy(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case string:
		return x != ""
	case float64:
		return x != 0
	case map[string]any:
		return len(x) > 0
	case []any:
		return len(x) > 0
	default:
		return true
	}
}
