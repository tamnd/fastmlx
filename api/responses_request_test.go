// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

// Fixtures in testdata/parity/responses_request.json are captured from the
// reference convert_responses_input_to_messages and convert_responses_tools.
// Both produce structures the chat-template layer consumes, where object key
// order is not contractual, so the comparison is structural (canonicalized
// JSON). Input items are stored in their post-validation form so the port sees
// the same normalized values the reference function consumed (notably a
// list/dict function_call_output that the model serializes to a JSON string).
// Every captured function_call carries an explicit call_id, so no random ids
// enter the comparison.

type responsesRequestFixtures struct {
	InputCases []struct {
		Label            string          `json:"label"`
		Input            json.RawMessage `json:"input"`
		Instructions     *string         `json:"instructions"`
		PreviousMessages json.RawMessage `json:"previous_messages"`
		Result           json.RawMessage `json:"result"`
	} `json:"input_cases"`
	ToolCases []struct {
		Label  string          `json:"label"`
		Tools  json.RawMessage `json:"tools"`
		Result json.RawMessage `json:"result"`
	} `json:"tool_cases"`
}

func loadResponsesRequestFixtures(t testing.TB) responsesRequestFixtures {
	t.Helper()
	data, err := os.ReadFile("testdata/parity/responses_request.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fx responsesRequestFixtures
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("decode fixtures: %v", err)
	}
	if len(fx.InputCases) == 0 || len(fx.ToolCases) == 0 {
		t.Fatal("missing fixture cases")
	}
	return fx
}

// parseMessageList decodes a JSON array of message objects into jval messages,
// returning nil for a JSON null or absent value.
func parseMessageList(t *testing.T, raw json.RawMessage) []jval {
	t.Helper()
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	v, ok := parseOrdered(string(raw))
	if !ok || v.kind != kindArray {
		t.Fatalf("message list did not parse as array: %s", raw)
	}
	return v.arr
}

func TestConvertResponsesInputToMessagesParity(t *testing.T) {
	for _, c := range loadResponsesRequestFixtures(t).InputCases {
		t.Run(c.Label, func(t *testing.T) {
			input := jval{kind: kindNull}
			if len(c.Input) > 0 && string(c.Input) != "null" {
				v, ok := parseOrdered(string(c.Input))
				if !ok {
					t.Fatalf("input did not parse: %s", c.Input)
				}
				input = v
			}
			instructions := ""
			if c.Instructions != nil {
				instructions = *c.Instructions
			}
			prev := parseMessageList(t, c.PreviousMessages)

			result := ConvertResponsesInputToMessages(input, instructions, prev)

			got := canonJSON(t, []byte(jval{kind: kindArray, arr: result}.dumpASCII()))
			want := canonJSON(t, c.Result)
			if got != want {
				t.Errorf("case %q\n got  %s\n want %s", c.Label, got, want)
			}
		})
	}
}

func TestConvertResponsesToolsParity(t *testing.T) {
	for _, c := range loadResponsesRequestFixtures(t).ToolCases {
		t.Run(c.Label, func(t *testing.T) {
			toolsVal, ok := parseOrdered(string(c.Tools))
			if !ok || toolsVal.kind != kindArray {
				t.Fatalf("tools did not parse as array: %s", c.Tools)
			}
			result := ConvertResponsesTools(toolsVal.arr)

			var want string
			if string(c.Result) == "null" {
				want = "null"
			} else {
				want = canonJSON(t, c.Result)
			}
			if result == nil {
				if want != "null" {
					t.Errorf("case %q got nil, want %s", c.Label, want)
				}
				return
			}
			marshaled, err := json.Marshal(result)
			if err != nil {
				t.Fatalf("marshal result: %v", err)
			}
			got := canonJSON(t, marshaled)
			if got != want {
				t.Errorf("case %q\n got  %s\n want %s", c.Label, got, want)
			}
		})
	}
}

func TestConvertResponsesInputMintsCallID(t *testing.T) {
	// A function_call with no call_id or id must still produce a tool call with
	// a freshly minted call_<hex> id rather than an empty one.
	input := jval{kind: kindArray, arr: []jval{
		jobj("type", jstr("function_call"), "name", jstr("f"), "arguments", jstr("{}")),
	}}
	result := ConvertResponsesInputToMessages(input, "", nil)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	calls, ok := result[0].getField("tool_calls")
	if !ok || calls.kind != kindArray || len(calls.arr) != 1 {
		t.Fatalf("expected one tool call, got %v", result[0])
	}
	id := calls.arr[0].getString("id")
	if len(id) < 6 || id[:5] != "call_" {
		t.Errorf("minted id %q is not call_<hex>", id)
	}
}

func BenchmarkConvertResponsesInputToMessages(b *testing.B) {
	input := jval{kind: kindArray, arr: []jval{
		jobj("role", jstr("user"), "content", jstr("weather in Paris?")),
		jobj("type", jstr("reasoning"), "summary", jval{kind: kindArray, arr: []jval{
			jobj("type", jstr("summary_text"), "text", jstr("need a tool")),
		}}),
		jobj("type", jstr("function_call"), "call_id", jstr("call_x"),
			"name", jstr("get_weather"), "arguments", jstr(`{"city": "Paris"}`)),
		jobj("type", jstr("function_call_output"), "call_id", jstr("call_x"),
			"output", jstr("sunny")),
		jobj("role", jstr("assistant"), "content", jstr("It is sunny.")),
	}}
	b.ReportAllocs()
	for b.Loop() {
		_ = ConvertResponsesInputToMessages(input, "be helpful", nil)
	}
}
