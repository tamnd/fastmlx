// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"testing"
)

// Fixtures in testdata/parity/responses_convert.json are captured from the
// reference build_*_output_item helpers and ResponseObject.model_dump_json with
// fixed sentinel ids and created_at. The wire form is contractual (a serialized
// HTTP body), so the comparison is byte-exact: the port builds the same object
// with the same sentinel ids and serializes with dumpCompact.

type responsesConvertCase struct {
	Label string `json:"label"`
	Input struct {
		Text            string `json:"text"`
		Reasoning       string `json:"reasoning"`
		NativeReasoning bool   `json:"native_reasoning"`
		ToolCalls       []struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
			CallID    string `json:"call_id"`
		} `json:"tool_calls"`
		PromptTokens     int             `json:"prompt_tokens"`
		CompletionTokens int             `json:"completion_tokens"`
		ReasoningTokens  int             `json:"reasoning_tokens"`
		CachedTokens     int             `json:"cached_tokens"`
		Temperature      json.RawMessage `json:"temperature"`
		TopP             json.RawMessage `json:"top_p"`
		MaxOutputTokens  json.RawMessage `json:"max_output_tokens"`
		ToolChoice       json.RawMessage `json:"tool_choice"`
		Tools            json.RawMessage `json:"tools"`
		PreviousResponse *string         `json:"previous_response_id"`
	} `json:"input"`
	Raw string `json:"raw"`
}

func loadResponsesConvertCases(t testing.TB) []responsesConvertCase {
	t.Helper()
	data, err := os.ReadFile("testdata/parity/responses_convert.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fx struct {
		Cases []responsesConvertCase `json:"cases"`
	}
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("decode fixtures: %v", err)
	}
	if len(fx.Cases) == 0 {
		t.Fatal("no fixture cases")
	}
	return fx.Cases
}

func rawToJval(t *testing.T, raw json.RawMessage) jval {
	t.Helper()
	if len(raw) == 0 || string(raw) == "null" {
		return jnull()
	}
	v, ok := parseOrdered(string(raw))
	if !ok {
		t.Fatalf("could not parse %s", raw)
	}
	return v
}

func (c responsesConvertCase) toInput(t *testing.T) ResponsesResponseInput {
	t.Helper()
	in := c.Input
	calls := make([]ToolCall, 0, len(in.ToolCalls))
	for _, tc := range in.ToolCalls {
		calls = append(calls, ToolCall{
			ID:       tc.CallID,
			Function: FunctionCall{Name: tc.Name, Arguments: tc.Arguments},
		})
	}
	var tools []jval
	if tv := rawToJval(t, in.Tools); tv.kind == kindArray {
		tools = tv.arr
	}
	prev := ""
	if in.PreviousResponse != nil {
		prev = *in.PreviousResponse
	}
	return ResponsesResponseInput{
		Model:              "m",
		Text:               in.Text,
		Reasoning:          in.Reasoning,
		NativeReasoning:    in.NativeReasoning,
		ToolCalls:          calls,
		PromptTokens:       in.PromptTokens,
		CompletionTokens:   in.CompletionTokens,
		ReasoningTokens:    in.ReasoningTokens,
		CachedTokens:       in.CachedTokens,
		Temperature:        rawToJval(t, in.Temperature),
		TopP:               rawToJval(t, in.TopP),
		MaxOutputTokens:    rawToJval(t, in.MaxOutputTokens),
		ToolChoice:         rawToJval(t, in.ToolChoice),
		Tools:              tools,
		PreviousResponseID: prev,
	}
}

func TestConvertInternalToResponsesResponseParity(t *testing.T) {
	for _, c := range loadResponsesConvertCases(t) {
		t.Run(c.Label, func(t *testing.T) {
			in := c.toInput(t)
			ids := responsesItemIDs{reasoning: "rs_PARITY", message: "msg_PARITY"}
			for i := range in.ToolCalls {
				ids.functionCall = append(ids.functionCall, fmt.Sprintf("fc_PARITY_%d", i))
			}
			got := buildResponseObject("resp_PARITY", 0, ids, in).dumpCompact()
			if got != c.Raw {
				t.Errorf("case %q\n got  %s\n want %s", c.Label, got, c.Raw)
			}
		})
	}
}

func TestConvertInternalToResponsesResponseMintsIDs(t *testing.T) {
	in := ResponsesResponseInput{
		Model:     "m",
		Text:      "hi",
		Reasoning: "thinking",
		ToolCalls: []ToolCall{{ID: "call_1", Function: FunctionCall{Name: "f", Arguments: "{}"}}},
	}
	out := ConvertInternalToResponsesResponse(1700000000, in)
	v, ok := parseOrdered(out)
	if !ok {
		t.Fatalf("output did not parse: %s", out)
	}
	respID := v.getString("id")
	if !regexp.MustCompile(`^resp_[0-9a-f]{24}$`).MatchString(respID) {
		t.Errorf("response id %q is not resp_<24hex>", respID)
	}
	if ca, _ := v.getField("created_at"); ca.s != "1700000000" {
		t.Errorf("created_at = %q, want 1700000000", ca.s)
	}
	output, _ := v.getField("output")
	if output.kind != kindArray {
		t.Fatalf("output is not an array")
	}
	// Native reasoning is off here, so output is message + one function_call.
	if len(output.arr) != 2 {
		t.Fatalf("expected 2 output items, got %d", len(output.arr))
	}
	msgID := output.arr[0].getString("id")
	if !regexp.MustCompile(`^msg_[0-9a-f]{24}$`).MatchString(msgID) {
		t.Errorf("message id %q is not msg_<24hex>", msgID)
	}
	fcID := output.arr[1].getString("id")
	if !regexp.MustCompile(`^fc_[0-9a-f]{8}$`).MatchString(fcID) {
		t.Errorf("function_call id %q is not fc_<8hex>", fcID)
	}
}

func BenchmarkConvertInternalToResponsesResponse(b *testing.B) {
	in := ResponsesResponseInput{
		Model:            "m",
		Text:             "It is sunny in Paris.",
		Reasoning:        "the user asked about weather",
		NativeReasoning:  true,
		ToolCalls:        []ToolCall{{ID: "call_1", Function: FunctionCall{Name: "get_weather", Arguments: `{"city": "Paris"}`}}},
		PromptTokens:     20,
		CompletionTokens: 12,
		ReasoningTokens:  5,
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = ConvertInternalToResponsesResponse(1700000000, in)
	}
}
