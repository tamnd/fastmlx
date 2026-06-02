// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "encoding/json"

// This file is the exported boundary between the Anthropic/Responses request
// conversions (which work on the unexported order-preserving jval model) and the
// routes layer. It turns raw request pieces into a flat InternalMessage list and
// tool definitions the prompt builder consumes. Content is flattened to text
// here; multimodal parts collapse to their text at this boundary, matching the
// chat-completions path.

// InternalMessage is one chat turn after request conversion, flattened to the
// fields the prompt builder uses.
type InternalMessage struct {
	Role             string
	Content          string
	ReasoningContent string
	ToolCallID       string
}

// AnthropicMessage is one entry of an Anthropic /v1/messages request, decoded
// only as far as its role and raw content (a string or a block list).
type AnthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// flattenContent reduces an internal-message content value to plain text: a
// string passes through; a block list concatenates the text of its text parts.
func flattenContent(content jval) string {
	switch content.kind {
	case kindString:
		return content.s
	case kindArray:
		var b []byte
		for _, part := range content.arr {
			if part.kind == kindObject && part.getString("type") == "text" {
				b = append(b, part.getString("text")...)
			}
		}
		return string(b)
	default:
		return ""
	}
}

// flattenInternal turns the jval internal messages the converters produce into
// the exported flat shape.
func flattenInternal(msgs []jval) []InternalMessage {
	out := make([]InternalMessage, 0, len(msgs))
	for _, m := range msgs {
		content := jval{kind: kindNull}
		if c, ok := m.getField("content"); ok {
			content = c
		}
		out = append(out, InternalMessage{
			Role:             m.getString("role"),
			Content:          flattenContent(content),
			ReasoningContent: m.getString("reasoning_content"),
			ToolCallID:       m.getString("tool_call_id"),
		})
	}
	return out
}

// AnthropicRequestToEngine converts the pieces of an Anthropic /v1/messages
// request into flat internal messages and tool definitions. The system value is
// the raw `system` field (a string or a text-block list); messages are the raw
// turns; tools are the advertised Anthropic tools (server-side tools are
// dropped).
func AnthropicRequestToEngine(system json.RawMessage, messages []AnthropicMessage, tools []AnthropicTool, opts AnthropicConvertOptions) ([]InternalMessage, []Tool) {
	var sys jval = jval{kind: kindNull}
	if len(system) > 0 && string(system) != "null" {
		if v, ok := parseOrdered(string(system)); ok {
			sys = v
		}
	}
	in := make([]AnthropicInMessage, 0, len(messages))
	for _, m := range messages {
		content := jval{kind: kindNull}
		if len(m.Content) > 0 && string(m.Content) != "null" {
			if v, ok := parseOrdered(string(m.Content)); ok {
				content = v
			}
		}
		in = append(in, AnthropicInMessage{Role: m.Role, Content: content})
	}
	internal := convertAnthropicToInternal(sys, in, opts)
	return flattenInternal(internal), ConvertAnthropicToolsToInternal(tools)
}

// ResponsesRequestToEngine converts the pieces of a Responses /v1/responses
// request into flat internal messages and tool definitions. input is the raw
// `input` field (a string or an item list), instructions the system prompt, and
// tools the raw Responses tool definitions.
func ResponsesRequestToEngine(input json.RawMessage, instructions string, tools []json.RawMessage) ([]InternalMessage, []Tool) {
	in := jval{kind: kindNull}
	if len(input) > 0 && string(input) != "null" {
		if v, ok := parseOrdered(string(input)); ok {
			in = v
		}
	}
	internal := ConvertResponsesInputToMessages(in, instructions, nil)

	toolVals := make([]jval, 0, len(tools))
	for _, raw := range tools {
		if v, ok := parseOrdered(string(raw)); ok {
			toolVals = append(toolVals, v)
		}
	}
	return flattenInternal(internal), ConvertResponsesTools(toolVals)
}

// ResponsesResult carries the route-level inputs for a non-stream Responses
// response: the generation text and token counts plus the request echo fields in
// JSON-friendly form. Temperature, TopP, MaxOutputTokens, and ToolChoice are raw
// request JSON so the exact value round-trips without reformatting; an empty
// value means the request omitted the field. Tools is the raw tool list echoed
// back verbatim.
type ResponsesResult struct {
	Model              string
	Text               string
	PromptTokens       int
	CompletionTokens   int
	CachedTokens       int
	Temperature        json.RawMessage
	TopP               json.RawMessage
	MaxOutputTokens    json.RawMessage
	ToolChoice         json.RawMessage
	Tools              []json.RawMessage
	PreviousResponseID string
}

// rawToJvalOrUnset parses a raw request value, returning the zero jval (kind 0,
// "unset") when the field was absent so the response assembler applies its
// default.
func rawToJvalOrUnset(raw json.RawMessage) jval {
	if len(raw) == 0 {
		return jval{}
	}
	if v, ok := parseOrdered(string(raw)); ok {
		return v
	}
	return jval{}
}

// parseRawJvalList parses a raw JSON list into jval values, skipping any that
// fail to parse.
func parseRawJvalList(raws []json.RawMessage) []jval {
	out := make([]jval, 0, len(raws))
	for _, raw := range raws {
		if v, ok := parseOrdered(string(raw)); ok {
			out = append(out, v)
		}
	}
	return out
}

// BuildResponsesResponse assembles a non-stream /v1/responses body from the
// generation result and the echoed request fields, minting fresh ids. The wire
// form matches the reference's pydantic model_dump_json.
func BuildResponsesResponse(createdAt int, res ResponsesResult) string {
	tools := parseRawJvalList(res.Tools)
	in := ResponsesResponseInput{
		Model:              res.Model,
		Text:               res.Text,
		PromptTokens:       res.PromptTokens,
		CompletionTokens:   res.CompletionTokens,
		CachedTokens:       res.CachedTokens,
		Temperature:        rawToJvalOrUnset(res.Temperature),
		TopP:               rawToJvalOrUnset(res.TopP),
		MaxOutputTokens:    rawToJvalOrUnset(res.MaxOutputTokens),
		ToolChoice:         rawToJvalOrUnset(res.ToolChoice),
		Tools:              tools,
		PreviousResponseID: res.PreviousResponseID,
	}
	return ConvertInternalToResponsesResponse(createdAt, in)
}

// ResponsesStreamInit carries the ids and echoed request fields a streamed
// /v1/responses needs. The sampling and tool_choice echoes are raw request JSON
// so they round-trip without reformatting; Tools is the raw tool list.
type ResponsesStreamInit struct {
	ResponseID         string
	MessageID          string
	Model              string
	CreatedAt          int
	Temperature        json.RawMessage
	TopP               json.RawMessage
	MaxOutputTokens    json.RawMessage
	ToolChoice         json.RawMessage
	Tools              []json.RawMessage
	PreviousResponseID string
}

// ResponsesStreamWriter produces the Responses SSE event sequence for a
// text-only generation (the path the mock backend drives): no native reasoning
// and no tool calls, so the single message item opens immediately at output
// index 0. It tracks the running sequence number across calls. Reasoning and
// function-call events exist in the event layer for when a backend emits them.
type ResponsesStreamWriter struct {
	init  streamResponseInit
	msgID string
	seq   int
}

// NewResponsesStreamWriter builds a writer over the response/message ids and the
// echoed request fields.
func NewResponsesStreamWriter(in ResponsesStreamInit) *ResponsesStreamWriter {
	return &ResponsesStreamWriter{
		init: streamResponseInit{
			ID:                 in.ResponseID,
			Model:              in.Model,
			CreatedAt:          in.CreatedAt,
			Status:             "in_progress",
			Temperature:        rawToJvalOrUnset(in.Temperature),
			TopP:               rawToJvalOrUnset(in.TopP),
			MaxOutputTokens:    rawToJvalOrUnset(in.MaxOutputTokens),
			ToolChoice:         rawToJvalOrUnset(in.ToolChoice),
			Tools:              parseRawJvalList(in.Tools),
			PreviousResponseID: in.PreviousResponseID,
		},
		msgID: in.MessageID,
	}
}

// Start returns the opening events: response.created, response.in_progress, the
// message output_item.added, and its content_part.added.
func (w *ResponsesStreamWriter) Start() []string {
	initial := buildStreamInitial(w.init)
	out := make([]string, 0, 4)
	w.seq++
	out = append(out, evCreated(w.seq, initial))
	w.seq++
	out = append(out, evInProgress(w.seq, initial))
	w.seq++
	out = append(out, evOutputItemAdded(w.seq, 0, messageItemInProgress(w.msgID)))
	w.seq++
	out = append(out, evContentPartAdded(w.seq, w.msgID, 0, 0, outputTextPart("")))
	return out
}

// TextDelta returns one output_text.delta event for a chunk of generated text.
func (w *ResponsesStreamWriter) TextDelta(delta string) string {
	w.seq++
	return evOutputTextDelta(w.seq, w.msgID, 0, 0, delta)
}

// Finish returns the closing events: output_text.done, content_part.done, the
// message output_item.done, and response.completed with final usage.
func (w *ResponsesStreamWriter) Finish(finalText string, promptTokens, completionTokens, cachedTokens int) []string {
	out := make([]string, 0, 4)
	w.seq++
	out = append(out, evOutputTextDone(w.seq, w.msgID, 0, 0, finalText))
	w.seq++
	out = append(out, evContentPartDone(w.seq, w.msgID, 0, 0, outputTextPart(finalText)))
	w.seq++
	out = append(out, evOutputItemDone(w.seq, 0, messageItemCompleted(w.msgID, finalText)))
	usage := buildStreamUsage(promptTokens, completionTokens, cachedTokens, 0)
	final := buildStreamFinal(w.init, []jval{messageItemCompleted(w.msgID, finalText)}, usage)
	w.seq++
	out = append(out, evCompleted(w.seq, final))
	return out
}

// Failed returns a response.failed event carrying the in-progress object with
// its status flipped to "failed".
func (w *ResponsesStreamWriter) Failed() string {
	f := w.init
	f.Status = "failed"
	w.seq++
	return evFailed(w.seq, buildStreamInitial(f))
}
