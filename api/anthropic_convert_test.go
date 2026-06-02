// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"regexp"
	"testing"
)

// Fixtures in testdata/parity/anthropic_convert.json are captured from the
// Python reference conversion functions. Response fixtures store the
// model_dump_json() output with the random message id replaced by the sentinel
// "msg_PARITY" and the thinking signature scrubbed to "fastmlx-reasoning", so
// the byte-exact comparison ignores only what is genuinely non-deterministic or
// renamed. Tool fixtures store the internal tool structure for a structural
// compare.

type convertFixtures struct {
	Responses []struct {
		Label string `json:"label"`
		Raw   string `json:"raw"`
	} `json:"responses"`
	Tools []struct {
		Label  string          `json:"label"`
		Result json.RawMessage `json:"result"`
	} `json:"tools"`
}

func loadConvertFixtures(t testing.TB) convertFixtures {
	t.Helper()
	data, err := os.ReadFile("testdata/parity/anthropic_convert.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fx convertFixtures
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("decode fixtures: %v", err)
	}
	if len(fx.Responses) == 0 || len(fx.Tools) == 0 {
		t.Fatal("fixtures missing a group")
	}
	return fx
}

func tcArg(id, name, args string) ToolCall {
	return ToolCall{ID: id, Type: "function", Function: FunctionCall{Name: name, Arguments: args}}
}

// responseByLabel reproduces the exact inputs the capture script used.
func responseByLabel(label string) (AnthropicResponseInput, bool) {
	switch label {
	case "text_only":
		return AnthropicResponseInput{Text: "Hello, world", Model: "claude-test", PromptTokens: 10, CompletionTokens: 3, FinishReason: "stop"}, true
	case "empty_text":
		return AnthropicResponseInput{Text: "", Model: "m", PromptTokens: 5, CompletionTokens: 0, FinishReason: "stop"}, true
	case "whitespace_text_becomes_empty_block":
		return AnthropicResponseInput{Text: "   ", Model: "m", PromptTokens: 5, CompletionTokens: 0, FinishReason: "stop"}, true
	case "length_finish":
		return AnthropicResponseInput{Text: "truncated", Model: "m", PromptTokens: 8, CompletionTokens: 100, FinishReason: "length"}, true
	case "thinking_then_text":
		return AnthropicResponseInput{Text: "The answer is 4", Model: "m", PromptTokens: 10, CompletionTokens: 20, FinishReason: "stop", Thinking: "let me add 2 and 2"}, true
	case "thinking_only":
		return AnthropicResponseInput{Text: "", Model: "m", PromptTokens: 10, CompletionTokens: 20, FinishReason: "stop", Thinking: "reasoning with no visible answer"}, true
	case "unicode_text":
		return AnthropicResponseInput{Text: "café 日本語 ☃", Model: "m", PromptTokens: 10, CompletionTokens: 5, FinishReason: "stop"}, true
	case "single_tool_call":
		return AnthropicResponseInput{Text: "", Model: "m", PromptTokens: 10, CompletionTokens: 15, FinishReason: "tool_calls", ToolCalls: []ToolCall{tcArg("toolu_1", "get_weather", `{"city": "Paris"}`)}}, true
	case "text_and_tool_call":
		return AnthropicResponseInput{Text: "Let me check.", Model: "m", PromptTokens: 10, CompletionTokens: 15, FinishReason: "tool_calls", ToolCalls: []ToolCall{tcArg("toolu_2", "search", `{"q": "café", "limit": 5}`)}}, true
	case "multi_tool_calls":
		return AnthropicResponseInput{Text: "", Model: "m", PromptTokens: 10, CompletionTokens: 15, FinishReason: "tool_calls", ToolCalls: []ToolCall{
			tcArg("toolu_a", "f", `{"x": 1}`),
			tcArg("toolu_b", "g", `{"y": [1, 2, 3], "z": {"k": true}}`),
		}}, true
	case "thinking_text_and_tool":
		return AnthropicResponseInput{Text: "Calling tool", Model: "m", PromptTokens: 10, CompletionTokens: 15, FinishReason: "tool_calls", Thinking: "I should call the tool", ToolCalls: []ToolCall{tcArg("toolu_d", "act", `{"do": "it"}`)}}, true
	case "cache_control_split":
		return AnthropicResponseInput{Text: "cached", Model: "m", PromptTokens: 100, CompletionTokens: 10, FinishReason: "stop", CachedTokens: 30, RequestUsesCacheControl: true}, true
	case "cache_control_over_clamp":
		return AnthropicResponseInput{Text: "cached", Model: "m", PromptTokens: 50, CompletionTokens: 7, FinishReason: "length", CachedTokens: 80, RequestUsesCacheControl: true}, true
	case "cache_hit_without_optin_ignored":
		return AnthropicResponseInput{Text: "x", Model: "m", PromptTokens: 40, CompletionTokens: 2, FinishReason: "stop", CachedTokens: 40, RequestUsesCacheControl: false}, true
	}
	return AnthropicResponseInput{}, false
}

func TestConvertInternalToAnthropicResponseParity(t *testing.T) {
	for _, fx := range loadConvertFixtures(t).Responses {
		t.Run(fx.Label, func(t *testing.T) {
			in, ok := responseByLabel(fx.Label)
			if !ok {
				t.Fatalf("no Go input wired for label %q", fx.Label)
			}
			got := buildAnthropicResponse("msg_PARITY", in).dumpCompact()
			if got != fx.Raw {
				t.Errorf("response %q\n got  %s\n want %s", fx.Label, got, fx.Raw)
			}
		})
	}
}

var msgIDRe = regexp.MustCompile(`^msg_[0-9a-f]{24}$`)

func TestConvertInternalToAnthropicResponseMintsID(t *testing.T) {
	in, _ := responseByLabel("text_only")
	body := ConvertInternalToAnthropicResponse(in)
	var got struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal([]byte(body), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !msgIDRe.MatchString(got.ID) {
		t.Errorf("minted id %q does not match msg_<24 hex>", got.ID)
	}
	seen := map[string]bool{}
	for range 100 {
		var r struct {
			ID string `json:"id"`
		}
		_ = json.Unmarshal([]byte(ConvertInternalToAnthropicResponse(in)), &r)
		if seen[r.ID] {
			t.Fatalf("duplicate minted id %q", r.ID)
		}
		seen[r.ID] = true
	}
}

// toolsByLabel reproduces the capture script's Anthropic tool inputs.
func toolsByLabel(label string) ([]AnthropicTool, bool) {
	switch label {
	case "none":
		return nil, true
	case "empty":
		return []AnthropicTool{}, true
	case "single_user_tool":
		return []AnthropicTool{{
			Name:        "get_weather",
			Description: new("Get weather"),
			InputSchema: json.RawMessage(`{"type": "object", "properties": {"city": {"type": "string"}}}`),
		}}, true
	case "no_description":
		return []AnthropicTool{{
			Name:        "ping",
			InputSchema: json.RawMessage(`{"type": "object"}`),
		}}, true
	case "null_input_schema_becomes_empty":
		return []AnthropicTool{{
			Name:        "noargs",
			Description: new("d"),
			InputSchema: json.RawMessage(`{}`),
		}}, true
	case "drops_server_side":
		return []AnthropicTool{
			{Name: "web_search", Type: new("web_search_20250305")},
			{Name: "real", Description: new("kept"), InputSchema: json.RawMessage(`{"type": "object"}`)},
		}, true
	case "all_server_side_returns_none":
		return []AnthropicTool{
			{Name: "bash", Type: new("bash_20250124")},
			{Name: "editor", Type: new("text_editor_20250728")},
		}, true
	case "dict_input":
		return []AnthropicTool{{
			Name:        "from_dict",
			Description: new("dd"),
			InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
		}}, true
	}
	return nil, false
}

// canonJSON normalizes a JSON value (sorting object keys) so two structurally
// equal values compare equal regardless of key order or spacing.
func canonJSON(t *testing.T, raw []byte) string {
	t.Helper()
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("canon decode %q: %v", raw, err)
	}
	out, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("canon encode: %v", err)
	}
	return string(out)
}

func TestConvertAnthropicToolsToInternalParity(t *testing.T) {
	for _, fx := range loadConvertFixtures(t).Tools {
		t.Run(fx.Label, func(t *testing.T) {
			tools, ok := toolsByLabel(fx.Label)
			if !ok {
				t.Fatalf("no Go input wired for label %q", fx.Label)
			}
			got := ConvertAnthropicToolsToInternal(tools)

			// The fixture result is either null or an array of internal tools.
			if string(fx.Result) == "null" {
				if got != nil {
					t.Errorf("tools %q: got %d tools, want nil", fx.Label, len(got))
				}
				return
			}

			var want []struct {
				Type     string `json:"type"`
				Function struct {
					Name        string          `json:"name"`
					Description *string         `json:"description"`
					Parameters  json.RawMessage `json:"parameters"`
				} `json:"function"`
			}
			if err := json.Unmarshal(fx.Result, &want); err != nil {
				t.Fatalf("decode want: %v", err)
			}
			if len(got) != len(want) {
				t.Fatalf("tools %q: got %d, want %d", fx.Label, len(got), len(want))
			}
			for i := range want {
				if got[i].Type != want[i].Type {
					t.Errorf("tool %d type: got %q want %q", i, got[i].Type, want[i].Type)
				}
				if got[i].Function.Name != want[i].Function.Name {
					t.Errorf("tool %d name: got %q want %q", i, got[i].Function.Name, want[i].Function.Name)
				}
				// A null description in the reference maps to the empty string
				// in the internal tool.
				wantDesc := ""
				if want[i].Function.Description != nil {
					wantDesc = *want[i].Function.Description
				}
				if got[i].Function.Description != wantDesc {
					t.Errorf("tool %d description: got %q want %q", i, got[i].Function.Description, wantDesc)
				}
				gotParams := canonJSON(t, got[i].Function.Parameters)
				wantParams := canonJSON(t, want[i].Function.Parameters)
				if gotParams != wantParams {
					t.Errorf("tool %d parameters: got %s want %s", i, gotParams, wantParams)
				}
			}
		})
	}
}

// TestToolCallArgumentsFallback covers the defensive branch the reference guards
// but cannot reach through its FunctionCall model: arguments that do not parse to
// a JSON object collapse to an empty input object.
func TestToolCallArgumentsFallback(t *testing.T) {
	for _, args := range []string{"not json", "", "[1,2,3]", "42", `"a string"`, "null"} {
		in := AnthropicResponseInput{
			Model: "m", PromptTokens: 1, CompletionTokens: 1, FinishReason: "tool_calls",
			ToolCalls: []ToolCall{tcArg("toolu_x", "f", args)},
		}
		body := buildAnthropicResponse("msg_PARITY", in).dumpCompact()
		want := `{"id":"msg_PARITY","type":"message","role":"assistant","model":"m",` +
			`"content":[{"type":"tool_use","id":"toolu_x","name":"f","input":{}}],` +
			`"stop_reason":"tool_use","stop_sequence":null,` +
			`"usage":{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}`
		if body != want {
			t.Errorf("args %q\n got  %s\n want %s", args, body, want)
		}
	}
}

func BenchmarkConvertInternalToAnthropicResponse(b *testing.B) {
	in := AnthropicResponseInput{
		Text: "The quick brown fox", Model: "m", PromptTokens: 100, CompletionTokens: 20,
		FinishReason: "tool_calls", Thinking: "reasoning here",
		ToolCalls: []ToolCall{tcArg("toolu_x", "search", `{"q": "weather", "limit": 5}`)},
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = buildAnthropicResponse("msg_bench", in).dumpCompact()
	}
}
