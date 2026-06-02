// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"strings"
)

// This file ports the non-streaming half of the Anthropic translation layer:
// turning Anthropic tool definitions into the internal function-tool shape, and
// turning internal generation output into an Anthropic MessagesResponse. Both
// are pure data transforms with no model toolkit involved.
//
// The endpoint serializes the response with pydantic's model_dump_json(), which
// is compact (no spaces after ',' and ':') and passes non-ASCII through as
// UTF-8. dumpCompact reproduces that exactly. The message id is a random value
// in the reference, so it is excluded from byte-exact comparison.

// serverSideToolTypePrefixes name the Anthropic server-side tool families that
// this server cannot execute locally. Tools whose "type" starts with one of
// these are accepted for SDK compatibility and then dropped before inference.
var serverSideToolTypePrefixes = []string{
	"web_search_",
	"code_execution_",
	"bash_",
	"text_editor_",
	"computer_",
}

// AnthropicTool is a tool definition in Anthropic's request shape. A
// user-defined tool carries input_schema; a server-side tool carries a
// versioned type (e.g. "web_search_20250305") and no input_schema. Description
// is a pointer so an absent field stays distinct from an empty string, matching
// the reference's Optional[str] default of None.
type AnthropicTool struct {
	Name        string          `json:"name"`
	Description *string         `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
	Type        *string         `json:"type,omitempty"`
}

// isServerSideTool reports whether the tool's type marks it as an Anthropic
// server-side tool that this server cannot run.
func isServerSideTool(t AnthropicTool) bool {
	if t.Type == nil {
		return false
	}
	for _, prefix := range serverSideToolTypePrefixes {
		if strings.HasPrefix(*t.Type, prefix) {
			return true
		}
	}
	return false
}

// ConvertAnthropicToolsToInternal maps Anthropic tools to the internal
// function-tool shape. Server-side tools are dropped. It returns nil when there
// are no executable tools (matching the reference's None), so callers can omit
// the tools field entirely.
func ConvertAnthropicToolsToInternal(tools []AnthropicTool) []Tool {
	if len(tools) == 0 {
		return nil
	}

	internal := make([]Tool, 0, len(tools))
	for _, t := range tools {
		if isServerSideTool(t) {
			continue
		}

		// input_schema or {}: a missing or null schema becomes an empty object.
		params := json.RawMessage("{}")
		if trimmed := strings.TrimSpace(string(t.InputSchema)); trimmed != "" && trimmed != "null" {
			params = t.InputSchema
		}

		desc := ""
		if t.Description != nil {
			desc = *t.Description
		}

		internal = append(internal, Tool{
			Type: "function",
			Function: FunctionDef{
				Name:        t.Name,
				Description: desc,
				Parameters:  params,
			},
		})
	}

	if len(internal) == 0 {
		return nil
	}
	return internal
}

// AnthropicResponseInput carries the internal generation output that becomes an
// Anthropic MessagesResponse.
type AnthropicResponseInput struct {
	Text                    string
	Model                   string
	PromptTokens            int
	CompletionTokens        int
	FinishReason            string
	ToolCalls               []ToolCall
	Thinking                string
	CachedTokens            int
	RequestUsesCacheControl bool
}

// newMessageID mints an Anthropic-style "msg_<24 hex>" identifier. The reference
// uses a random uuid suffix, so the id is never part of parity comparisons.
func newMessageID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return "msg_" + hex.EncodeToString(b[:])
}

// buildAnthropicResponse assembles the ordered response object for a fixed id.
// Content blocks are emitted in the reference order: thinking (when non-blank),
// then text (when non-blank), then one tool_use block per call. When nothing
// would be emitted, a single empty text block is added so content is never
// empty.
func buildAnthropicResponse(id string, in AnthropicResponseInput) jval {
	content := jval{kind: kindArray}

	if strings.TrimSpace(in.Thinking) != "" {
		content.arr = append(content.arr, jobj(
			"type", jstr("thinking"),
			"thinking", jstr(in.Thinking),
			"signature", jstr(thinkingSignature),
		))
	}

	if strings.TrimSpace(in.Text) != "" {
		content.arr = append(content.arr, jobj(
			"type", jstr("text"),
			"text", jstr(in.Text),
			"cache_control", jnull(),
		))
	}

	for _, tc := range in.ToolCalls {
		input, ok := parseOrdered(tc.Function.Arguments)
		if !ok || input.kind != kindObject {
			input = jval{kind: kindObject}
		}
		content.arr = append(content.arr, jobj(
			"type", jstr("tool_use"),
			"id", jstr(tc.ID),
			"name", jstr(tc.Function.Name),
			"input", input,
		))
	}

	if len(content.arr) == 0 {
		content.arr = append(content.arr, jobj(
			"type", jstr("text"),
			"text", jstr(""),
			"cache_control", jnull(),
		))
	}

	stopReason := MapFinishReasonToStopReason(in.FinishReason, len(in.ToolCalls) > 0)

	// The three input-side usage fields are a disjoint partition of the prompt
	// and only carry non-zero values when the request opted into caching. Absent
	// that signal the cache fields stay 0 regardless of any internal prefix-cache
	// hit, and input_tokens reports the full prompt.
	inputDisplay, cacheCreation, cacheRead := in.PromptTokens, 0, 0
	if in.RequestUsesCacheControl {
		cacheRead = max(0, min(in.CachedTokens, in.PromptTokens))
		cacheCreation = in.PromptTokens - cacheRead
		inputDisplay = 0
	}

	return jobj(
		"id", jstr(id),
		"type", jstr("message"),
		"role", jstr("assistant"),
		"model", jstr(in.Model),
		"content", content,
		"stop_reason", jstrOrNull(stopReason),
		"stop_sequence", jnull(),
		"usage", jobj(
			"input_tokens", jint(inputDisplay),
			"output_tokens", jint(in.CompletionTokens),
			"cache_creation_input_tokens", jint(cacheCreation),
			"cache_read_input_tokens", jint(cacheRead),
		),
	)
}

// ConvertInternalToAnthropicResponse mints a fresh message id and returns the
// MessagesResponse serialized exactly as the endpoint serializes it.
func ConvertInternalToAnthropicResponse(in AnthropicResponseInput) string {
	return buildAnthropicResponse(newMessageID(), in).dumpCompact()
}

// dumpCompact renders the value as pydantic's model_dump_json: keys in source
// order, "," and ":" separators with no spaces, and non-ASCII passed through.
func (v jval) dumpCompact() string {
	var b strings.Builder
	v.writeCompact(&b)
	return b.String()
}

func (v jval) writeCompact(b *strings.Builder) {
	switch v.kind {
	case kindObject:
		b.WriteByte('{')
		for i, kv := range v.obj {
			if i > 0 {
				b.WriteByte(',')
			}
			escapeString(b, kv.k, false)
			b.WriteByte(':')
			kv.v.writeCompact(b)
		}
		b.WriteByte('}')
	case kindArray:
		b.WriteByte('[')
		for i, item := range v.arr {
			if i > 0 {
				b.WriteByte(',')
			}
			item.writeCompact(b)
		}
		b.WriteByte(']')
	case kindString:
		escapeString(b, v.s, false)
	case kindNumber, kindBool:
		b.WriteString(v.s)
	case kindNull:
		b.WriteString("null")
	}
}
