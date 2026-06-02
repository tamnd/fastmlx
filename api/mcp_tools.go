// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "strings"

// This file ports the pure MCP <-> OpenAI tool schema conversions from the
// reference mcp/tools.py and the data shapes from mcp/types.py. These are
// GPU-free data transforms: turning MCP tool descriptors into OpenAI function
// definitions, merging them with user-supplied tools (user wins on a name
// clash), parsing a model tool call back into a (server, tool, arguments)
// triple, and formatting a tool result as a chat message. The live MCP client
// (stdio/SSE/HTTP transport and connection lifecycle) is a separate subsystem;
// only the schema translation lands here.

// MCPTool is a normalized tool descriptor discovered from an MCP server.
type MCPTool struct {
	ServerName  string
	Name        string
	Description string
	InputSchema jval // a JSON object, or the zero jval when none was provided
}

// FullName is the namespaced "server__tool" identifier exposed to the model.
func (t MCPTool) FullName() string { return t.ServerName + "__" + t.Name }

// defaultToolSchema is the empty object schema substituted when an MCP tool
// reports no input schema (mirrors `tool.input_schema or {...}`).
func defaultToolSchema() jval {
	return jobj("type", jstr("object"), "properties", jval{kind: kindObject})
}

// mcpToolToOpenAI converts one MCP tool to an OpenAI function definition. A
// missing or empty input schema falls back to the empty object schema, matching
// the reference truthiness check.
func mcpToolToOpenAI(t MCPTool) jval {
	params := t.InputSchema
	if params.kind != kindObject || len(params.obj) == 0 {
		params = defaultToolSchema()
	}
	return jobj(
		"type", jstr("function"),
		"function", jobj(
			"name", jstr(t.FullName()),
			"description", jstr(t.Description),
			"parameters", params,
		),
	)
}

// MCPToolsToOpenAI converts a list of MCP tools to OpenAI function definitions.
func MCPToolsToOpenAI(tools []MCPTool) []jval {
	out := make([]jval, 0, len(tools))
	for _, t := range tools {
		out = append(out, mcpToolToOpenAI(t))
	}
	return out
}

// OpenAICallToMCP parses an OpenAI tool call back into its MCP parts. arguments
// is decoded from the JSON string the model emitted (an unparseable string
// yields an empty object), or taken directly when already an object. A
// "server__tool" name splits on the first "__"; an un-namespaced name returns an
// empty server so the caller can resolve it.
func OpenAICallToMCP(toolCall jval) (serverName, toolName string, arguments jval) {
	function, _ := toolCall.getField("function")
	fullName := function.getString("name")

	arguments = jval{kind: kindObject}
	if argField, ok := function.getField("arguments"); ok {
		switch argField.kind {
		case kindString:
			if v, ok := parseOrdered(argField.s); ok && v.kind == kindObject {
				arguments = v
			}
		case kindObject:
			arguments = argField
		}
	}

	if before, after, found := strings.Cut(fullName, "__"); found {
		serverName, toolName = before, after
	} else {
		toolName = fullName
	}
	return serverName, toolName, arguments
}

// MergeTools merges MCP tools with user-supplied OpenAI tools. User tools take
// precedence on a name clash, overriding in place so the original ordering is
// preserved (MCP tools first in discovery order, then any new user tools).
func MergeTools(mcpTools []MCPTool, userTools []jval) []jval {
	order := make([]string, 0, len(mcpTools)+len(userTools))
	byName := make(map[string]jval)
	add := func(name string, tool jval) {
		if _, seen := byName[name]; !seen {
			order = append(order, name)
		}
		byName[name] = tool
	}
	for _, t := range mcpTools {
		add(t.FullName(), mcpToolToOpenAI(t))
	}
	for _, tool := range userTools {
		fn, _ := tool.getField("function")
		if name := fn.getString("name"); name != "" {
			add(name, tool)
		}
	}
	out := make([]jval, 0, len(order))
	for _, name := range order {
		out = append(out, byName[name])
	}
	return out
}

// MCPToolResult is the result of an MCP tool execution.
type MCPToolResult struct {
	ToolName     string
	Content      jval // string, object, or any JSON value
	IsError      bool
	ErrorMessage string
}

// ToMessage formats the result as an OpenAI tool-role message. An error becomes
// "Error: <message>"; string content passes through; any other content is
// JSON-encoded with json.dumps defaults (ASCII-escaped, spaced separators).
func (r MCPToolResult) ToMessage(toolCallID string) jval {
	var content string
	switch {
	case r.IsError:
		content = "Error: " + r.ErrorMessage
	case r.Content.kind == kindString:
		content = r.Content.s
	default:
		content = r.Content.dumpASCII()
	}
	return jobj(
		"role", jstr("tool"),
		"tool_call_id", jstr(toolCallID),
		"content", jstr(content),
	)
}

// FormatToolResult formats one tool result as a chat message.
func FormatToolResult(result MCPToolResult, toolCallID string) jval {
	return result.ToMessage(toolCallID)
}

// extractToolCalls returns the tool calls from a model response, or nil.
func extractToolCalls(response jval) []jval {
	choices, ok := response.getField("choices")
	if !ok || choices.kind != kindArray || len(choices.arr) == 0 {
		return nil
	}
	message, _ := choices.arr[0].getField("message")
	calls, ok := message.getField("tool_calls")
	if !ok || calls.kind != kindArray {
		return nil
	}
	return calls.arr
}

// HasToolCalls reports whether a model response contains any tool calls.
func HasToolCalls(response jval) bool {
	return len(extractToolCalls(response)) > 0
}
