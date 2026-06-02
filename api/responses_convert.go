// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"crypto/rand"
	"encoding/hex"
	"strings"
)

// This file ports the response-side OpenAI Responses API building: the pure
// output-item builders (build_message_output_item, build_function_call_output_item,
// build_reasoning_output_item, build_response_usage) and the ResponseObject
// assembly the non-stream /v1/responses handler returns. The wire form is
// pydantic model_dump_json with no exclude_none, so every declared field is
// emitted (null when unset); the order-preserving jval model reproduces it with
// dumpCompact (compact separators, non-ASCII passthrough).
//
// Only the reasoning-token count in build_response_usage is tokenizer-gated and
// supplied by the caller; everything here is a pure transform.

// newHexID mints "<prefix>_<n hex chars>" from crypto/rand, matching the
// reference generate_id. The reference uses a random uuid suffix, so these ids
// are never part of byte-exact parity comparisons.
func newHexID(prefix string, hexLen int) string {
	b := make([]byte, (hexLen+1)/2)
	_, _ = rand.Read(b)
	return prefix + "_" + hex.EncodeToString(b)[:hexLen]
}

func newResponseID() string     { return newHexID("resp", 24) }
func newReasoningID() string    { return newHexID("rs", 24) }
func newFunctionCallID() string { return newHexID("fc", 8) }

// buildMessageOutputItem builds a message-type output item carrying one
// output_text content block (with the always-present empty annotations list).
func buildMessageOutputItem(id, text string) jval {
	content := jval{kind: kindArray, arr: []jval{
		jobj("type", jstr("output_text"), "text", jstr(text), "annotations", jval{kind: kindArray}),
	}}
	return jobj(
		"type", jstr("message"),
		"id", jstr(id),
		"status", jstr("completed"),
		"role", jstr("assistant"),
		"content", content,
		"call_id", jnull(),
		"name", jnull(),
		"arguments", jnull(),
		"summary", jnull(),
	)
}

// buildFunctionCallOutputItem builds a function_call-type output item. arguments
// is carried as the raw JSON-string the model emitted, unparsed.
func buildFunctionCallOutputItem(id, name, arguments, callID string) jval {
	return jobj(
		"type", jstr("function_call"),
		"id", jstr(id),
		"status", jstr("completed"),
		"role", jnull(),
		"content", jnull(),
		"call_id", jstr(callID),
		"name", jstr(name),
		"arguments", jstr(arguments),
		"summary", jnull(),
	)
}

// buildReasoningOutputItem builds a reasoning-type output item carrying the full
// chain of thought in summary[0].text; the summary list is empty when the text
// is blank.
func buildReasoningOutputItem(id, reasoningText string) jval {
	summary := jval{kind: kindArray}
	if reasoningText != "" {
		summary.arr = []jval{jobj("type", jstr("summary_text"), "text", jstr(reasoningText))}
	}
	return jobj(
		"type", jstr("reasoning"),
		"id", jstr(id),
		"status", jstr("completed"),
		"role", jnull(),
		"content", jnull(),
		"call_id", jnull(),
		"name", jnull(),
		"arguments", jnull(),
		"summary", summary,
	)
}

// buildResponseUsage builds the Responses usage block. total_tokens is the sum
// of input and output (the reference model_post_init derives it the same way).
func buildResponseUsage(promptTokens, completionTokens, reasoningTokens, cachedTokens int) jval {
	return jobj(
		"input_tokens", jint(promptTokens),
		"output_tokens", jint(completionTokens),
		"total_tokens", jint(promptTokens+completionTokens),
		"input_tokens_details", jobj("cached_tokens", jint(cachedTokens)),
		"output_tokens_details", jobj("reasoning_tokens", jint(reasoningTokens)),
	)
}

// ResponsesResponseInput carries the generation result plus the request echo
// fields a non-stream Responses response repeats. Temperature, TopP, and
// MaxOutputTokens are passed through as parsed jval literals so the request's
// exact numeric form round-trips without float reformatting; pass jnull() when
// the request omitted them. ToolChoice defaults to the string "auto" when its
// kind is invalid (the zero jval). Tools is echoed verbatim (already in dumped
// form); an empty list is emitted when nil.
type ResponsesResponseInput struct {
	Model              string
	Text               string
	Reasoning          string
	NativeReasoning    bool
	ToolCalls          []ToolCall
	PromptTokens       int
	CompletionTokens   int
	ReasoningTokens    int
	CachedTokens       int
	Temperature        jval
	TopP               jval
	MaxOutputTokens    jval
	ToolChoice         jval
	Tools              []jval
	PreviousResponseID string
}

// buildResponseObject assembles the ordered ResponseObject for fixed ids and
// timestamp. Output items follow the reference order: an optional reasoning item
// (only when native reasoning is on and the text is non-blank), then always one
// message item, then one function_call item per tool call.
func buildResponseObject(id string, createdAt int, ids responsesItemIDs, in ResponsesResponseInput) jval {
	var output []jval
	reasoningText := strings.TrimSpace(in.Reasoning)
	if in.NativeReasoning && reasoningText != "" {
		output = append(output, buildReasoningOutputItem(ids.reasoning, reasoningText))
	}
	msgText := strings.TrimSpace(in.Text)
	output = append(output, buildMessageOutputItem(ids.message, msgText))
	for i, tc := range in.ToolCalls {
		callID := tc.ID
		if callID == "" {
			callID = newCallID()
		}
		output = append(output, buildFunctionCallOutputItem(ids.functionCall[i], tc.Function.Name, tc.Function.Arguments, callID))
	}

	usage := buildResponseUsage(in.PromptTokens, in.CompletionTokens, in.ReasoningTokens, in.CachedTokens)

	toolChoice := in.ToolChoice
	if toolChoice.kind == 0 {
		toolChoice = jstr("auto")
	}
	tools := jval{kind: kindArray}
	if len(in.Tools) > 0 {
		tools.arr = in.Tools
	}
	temperature := in.Temperature
	if temperature.kind == 0 {
		temperature = jnull()
	}
	topP := in.TopP
	if topP.kind == 0 {
		topP = jnull()
	}
	maxOut := in.MaxOutputTokens
	if maxOut.kind == 0 {
		maxOut = jnull()
	}

	return jobj(
		"id", jstr(id),
		"object", jstr("response"),
		"created_at", jint(createdAt),
		"model", jstr(in.Model),
		"status", jstr("completed"),
		"output", jval{kind: kindArray, arr: output},
		"usage", usage,
		"text", jnull(),
		"tool_choice", toolChoice,
		"tools", tools,
		"temperature", temperature,
		"top_p", topP,
		"max_output_tokens", maxOut,
		"previous_response_id", jstrOrNull(in.PreviousResponseID),
		"metadata", jval{kind: kindObject},
		"truncation", jnull(),
		"error", jnull(),
	)
}

// responsesItemIDs holds the minted output-item ids so the assembler can stay
// deterministic under test while minting fresh ids in production.
type responsesItemIDs struct {
	reasoning    string
	message      string
	functionCall []string
}

func mintItemIDs(in ResponsesResponseInput) responsesItemIDs {
	ids := responsesItemIDs{reasoning: newReasoningID(), message: newMessageID()}
	for range in.ToolCalls {
		ids.functionCall = append(ids.functionCall, newFunctionCallID())
	}
	return ids
}

// ConvertInternalToResponsesResponse builds a non-stream /v1/responses body,
// minting a fresh response id, output-item ids, and timestamp. The serialized
// form matches pydantic model_dump_json (compact, non-ASCII passthrough).
func ConvertInternalToResponsesResponse(createdAt int, in ResponsesResponseInput) string {
	return buildResponseObject(newResponseID(), createdAt, mintItemIDs(in), in).dumpCompact()
}
