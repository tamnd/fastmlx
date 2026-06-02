// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"strconv"
	"strings"
)

// Anthropic's Messages API streams a different SSE shape from OpenAI: every
// event carries an explicit "event: <type>" line, and the JSON payload is
// encoded with Python's default ensure_ascii=True, so non-ASCII text arrives
// \u-escaped. This file ports that event layer and the finish-reason mapping so
// the /v1/messages streaming path matches the reference byte for byte.
//
// The thinking-block signature is an opaque, required-non-empty placeholder;
// this server emits "fastmlx-reasoning" rather than naming any other project.

// thinkingSignature is the placeholder signature attached to streamed thinking
// blocks. Anthropic requires the field to be present and non-empty; the value
// itself is not interpreted by clients.
const thinkingSignature = "fastmlx-reasoning"

// MapFinishReasonToStopReason converts an internal finish reason to the
// Anthropic stop_reason. A response carrying tool calls always maps to
// "tool_use"; a nil finish reason maps to nil (returned as the empty string);
// anything unrecognized falls back to "end_turn".
func MapFinishReasonToStopReason(finishReason string, hasToolCalls bool) string {
	if hasToolCalls {
		return "tool_use"
	}
	switch finishReason {
	case "":
		return ""
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	default:
		return "end_turn"
	}
}

// formatSSEEvent renders one Anthropic SSE event: an "event:" line, a "data:"
// line carrying the JSON payload (ensure_ascii=True), and a blank separator.
func formatSSEEvent(eventType string, data jval) string {
	var b strings.Builder
	b.WriteString("event: ")
	b.WriteString(eventType)
	b.WriteString("\ndata: ")
	b.WriteString(data.dumpASCII())
	b.WriteString("\n\n")
	return b.String()
}

// CreateMessageStartEvent builds the message_start event that opens a stream.
func CreateMessageStartEvent(messageID, model string, inputTokens int) string {
	return formatSSEEvent("message_start", jobj(
		"type", jstr("message_start"),
		"message", jobj(
			"id", jstr(messageID),
			"type", jstr("message"),
			"role", jstr("assistant"),
			"model", jstr(model),
			"content", jval{kind: kindArray},
			"stop_reason", jnull(),
			"stop_sequence", jnull(),
			"usage", jobj(
				"input_tokens", jint(inputTokens),
				"output_tokens", jint(0),
			),
		),
	))
}

// CreateContentBlockStartEvent builds the content_block_start event for a text,
// tool_use, or thinking block (or a bare block for any other type). For a
// tool_use block, id and name carry the call identity.
func CreateContentBlockStartEvent(index int, blockType, id, name string) string {
	var block jval
	switch blockType {
	case "text":
		block = jobj("type", jstr("text"), "text", jstr(""))
	case "tool_use":
		block = jobj(
			"type", jstr("tool_use"),
			"id", jstr(id),
			"name", jstr(name),
			"input", jval{kind: kindObject},
		)
	case "thinking":
		block = jobj(
			"type", jstr("thinking"),
			"thinking", jstr(""),
			"signature", jstr(thinkingSignature),
		)
	default:
		block = jobj("type", jstr(blockType))
	}
	return formatSSEEvent("content_block_start", jobj(
		"type", jstr("content_block_start"),
		"index", jint(index),
		"content_block", block,
	))
}

// CreateThinkingDeltaEvent builds a content_block_delta carrying reasoning text.
func CreateThinkingDeltaEvent(index int, thinking string) string {
	return formatSSEEvent("content_block_delta", jobj(
		"type", jstr("content_block_delta"),
		"index", jint(index),
		"delta", jobj("type", jstr("thinking_delta"), "thinking", jstr(thinking)),
	))
}

// CreateTextDeltaEvent builds a content_block_delta carrying visible text.
func CreateTextDeltaEvent(index int, text string) string {
	return formatSSEEvent("content_block_delta", jobj(
		"type", jstr("content_block_delta"),
		"index", jint(index),
		"delta", jobj("type", jstr("text_delta"), "text", jstr(text)),
	))
}

// CreateInputJSONDeltaEvent builds a content_block_delta carrying a fragment of
// a tool call's input JSON.
func CreateInputJSONDeltaEvent(index int, partialJSON string) string {
	return formatSSEEvent("content_block_delta", jobj(
		"type", jstr("content_block_delta"),
		"index", jint(index),
		"delta", jobj("type", jstr("input_json_delta"), "partial_json", jstr(partialJSON)),
	))
}

// CreateContentBlockStopEvent builds the content_block_stop event.
func CreateContentBlockStopEvent(index int) string {
	return formatSSEEvent("content_block_stop", jobj(
		"type", jstr("content_block_stop"),
		"index", jint(index),
	))
}

// MessageDelta carries the optional inputs of a message_delta event. A nil
// pointer field means "not provided", matching the reference's None defaults.
type MessageDelta struct {
	StopReason              string
	OutputTokens            int
	StopSequence            string
	InputTokens             *int
	CachedTokens            int
	RequestUsesCacheControl bool
}

// CreateMessageDeltaEvent builds the message_delta event with final usage. When
// the request opted into cache control and input tokens are known, the count is
// split into Anthropic's disjoint triple (input 0, creation and read carry the
// rest); otherwise the cache fields are omitted.
func CreateMessageDeltaEvent(d MessageDelta) string {
	usage := jval{kind: kindObject}
	usage.obj = append(usage.obj, jkv{"output_tokens", jint(d.OutputTokens)})

	switch {
	case d.RequestUsesCacheControl && d.InputTokens != nil:
		in := *d.InputTokens
		cacheRead := max(0, min(d.CachedTokens, in))
		usage.obj = append(usage.obj,
			jkv{"input_tokens", jint(0)},
			jkv{"cache_creation_input_tokens", jint(in - cacheRead)},
			jkv{"cache_read_input_tokens", jint(cacheRead)},
		)
	case d.InputTokens != nil:
		usage.obj = append(usage.obj, jkv{"input_tokens", jint(*d.InputTokens)})
	}

	return formatSSEEvent("message_delta", jobj(
		"type", jstr("message_delta"),
		"delta", jobj(
			"stop_reason", jstrOrNull(d.StopReason),
			"stop_sequence", jstrOrNull(d.StopSequence),
		),
		"usage", usage,
	))
}

// CreateMessageStopEvent builds the terminal message_stop event.
func CreateMessageStopEvent() string {
	return formatSSEEvent("message_stop", jobj("type", jstr("message_stop")))
}

// CreatePingEvent builds a keep-alive ping event.
func CreatePingEvent() string {
	return formatSSEEvent("ping", jobj("type", jstr("ping")))
}

// CreateErrorEvent builds an error event with the given error type and message.
func CreateErrorEvent(errorType, message string) string {
	return formatSSEEvent("error", jobj(
		"type", jstr("error"),
		"error", jobj("type", jstr(errorType), "message", jstr(message)),
	))
}

// --- ordered JSON builders + ensure_ascii serialization ---

func jstr(s string) jval { return jval{kind: kindString, s: s} }
func jint(i int) jval    { return jval{kind: kindNumber, s: strconv.Itoa(i)} }
func jnull() jval        { return jval{kind: kindNull} }

// jstrOrNull renders a string value, or JSON null when the string is empty,
// matching the reference's Optional[str] fields that default to None.
func jstrOrNull(s string) jval {
	if s == "" {
		return jnull()
	}
	return jstr(s)
}

// jobj builds an object from alternating key, value arguments, preserving the
// argument order so the JSON keys come out in the order written.
func jobj(kv ...any) jval {
	o := jval{kind: kindObject}
	for i := 0; i+1 < len(kv); i += 2 {
		o.obj = append(o.obj, jkv{kv[i].(string), kv[i+1].(jval)})
	}
	return o
}

// dumpASCII renders the value as Python's json.dumps with default settings:
// keys in source order, ", "/": " separators, and ensure_ascii=True so every
// non-ASCII character is \u-escaped.
func (v jval) dumpASCII() string {
	var b strings.Builder
	v.writeASCII(&b)
	return b.String()
}

func (v jval) writeASCII(b *strings.Builder) {
	switch v.kind {
	case kindObject:
		b.WriteByte('{')
		for i, kv := range v.obj {
			if i > 0 {
				b.WriteString(", ")
			}
			escapeString(b, kv.k, true)
			b.WriteString(": ")
			kv.v.writeASCII(b)
		}
		b.WriteByte('}')
	case kindArray:
		b.WriteByte('[')
		for i, item := range v.arr {
			if i > 0 {
				b.WriteString(", ")
			}
			item.writeASCII(b)
		}
		b.WriteByte(']')
	case kindString:
		escapeString(b, v.s, true)
	case kindNumber, kindBool:
		b.WriteString(v.s)
	case kindNull:
		b.WriteString("null")
	}
}
