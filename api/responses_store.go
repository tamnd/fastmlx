// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "strings"

// This file ports the Responses API multi-turn store helpers: reconstructing a
// chat-template message history from a stored response's output items, and
// building the persisted state record for a response so a later turn can resume
// from previous_response_id. These are pure data transforms over the same
// order-preserving jval model the rest of the Responses translation uses; the
// store's disk persistence and id bookkeeping live with the admin/server layer.

// NormalizeResponseOutputToMessages turns a response's output items into
// assistant/tool-call history messages. Reasoning items attach as
// reasoning_content to the message or tool-call turn they precede, message items
// concatenate their output_text blocks, and consecutive function_call items
// accumulate into one synthesized assistant tool_calls message. System messages
// are consolidated to the front at the end.
func NormalizeResponseOutputToMessages(output jval) []jval {
	var messages []jval
	var pending []jval
	pendingReasoning := ""

	for _, item := range output.arr {
		switch item.getString("type") {
		case "reasoning":
			var parts []string
			if summary, ok := item.getField("summary"); ok && summary.kind == kindArray {
				for _, s := range summary.arr {
					if s.kind != kindObject {
						continue
					}
					if t := s.getString("text"); t != "" {
						parts = append(parts, t)
					}
				}
			}
			pendingReasoning = strings.Join(parts, "\n")
		case "message":
			pendingReasoning = flushPendingToolCalls(&messages, &pending, 0, pendingReasoning)
			var textParts []string
			if content, ok := item.getField("content"); ok && content.kind == kindArray {
				for _, block := range content.arr {
					if block.kind == kindObject && block.getString("type") == "output_text" {
						textParts = append(textParts, block.getString("text"))
					}
				}
			}
			role := item.getString("role")
			if role == "" {
				role = "assistant"
			}
			msg := jobj("role", jstr(role), "content", jstr(strings.Join(textParts, "\n")))
			if pendingReasoning != "" {
				msg = msg.setField("reasoning_content", jstr(pendingReasoning))
				pendingReasoning = ""
			}
			messages = append(messages, msg)
		case "function_call":
			callID := item.getString("call_id")
			if callID == "" {
				callID = newHexID("call", 8)
			}
			fn := jobj("name", jstr(item.getString("name")), "arguments", argumentsValue(item))
			pending = append(pending, jobj(
				"id", jstr(callID),
				"type", jstr("function"),
				"function", fn,
			))
		}
	}

	flushPendingToolCalls(&messages, &pending, 0, pendingReasoning)
	return consolidateSystemMessages(messages)
}

// argumentsValue resolves a function_call item's arguments the way the reference
// does: a missing field defaults to "{}", a string is parsed when it looks like
// JSON, and a non-string (already-parsed object or array) passes through whole.
func argumentsValue(item jval) jval {
	v, ok := item.getField("arguments")
	if !ok {
		return tryParseJSON("{}")
	}
	if v.kind != kindString {
		return v
	}
	return tryParseJSON(v.s)
}

// ConvertStoredResponseToMessages turns a stored public response or state record
// back into messages: a record that already carries output_messages returns them
// directly, otherwise the output items are normalized.
func ConvertStoredResponseToMessages(responseData jval) []jval {
	if om, ok := responseData.getField("output_messages"); ok {
		return append([]jval{}, om.arr...)
	}
	output, _ := responseData.getField("output")
	return NormalizeResponseOutputToMessages(output)
}

// BuildResponseStoreRecord builds the persisted state record for a response: its
// id and previous-response link, the input and output message lists, the full
// public response, and the creation timestamp. A missing previous_response_id
// records as null and a missing created_at as 0, matching the reference.
func BuildResponseStoreRecord(publicResponse jval, inputMessages, outputMessages []jval) jval {
	prev, ok := publicResponse.getField("previous_response_id")
	if !ok {
		prev = jval{kind: kindNull}
	}
	created, ok := publicResponse.getField("created_at")
	if !ok {
		created = jval{kind: kindNumber, s: "0"}
	}
	return jobj(
		"response_id", jstr(publicResponse.getString("id")),
		"previous_response_id", cloneJval(prev),
		"input_messages", jarrOf(inputMessages),
		"output_messages", jarrOf(outputMessages),
		"public_response", cloneJval(publicResponse),
		"created_at", cloneJval(created),
	)
}

// jarrOf wraps a slice of values into a jval array, deep-copying each element so
// the record does not alias caller-held messages (the reference deep-copies).
func jarrOf(items []jval) jval {
	arr := jval{kind: kindArray, arr: make([]jval, len(items))}
	for i, it := range items {
		arr.arr[i] = cloneJval(it)
	}
	return arr
}

// cloneJval returns a deep copy of v, mirroring the reference's copy.deepcopy on
// stored records.
func cloneJval(v jval) jval {
	switch v.kind {
	case kindObject:
		out := jval{kind: kindObject, obj: make([]jkv, len(v.obj))}
		for i, kv := range v.obj {
			out.obj[i] = jkv{kv.k, cloneJval(kv.v)}
		}
		return out
	case kindArray:
		out := jval{kind: kindArray, arr: make([]jval, len(v.arr))}
		for i, it := range v.arr {
			out.arr[i] = cloneJval(it)
		}
		return out
	default:
		return v
	}
}
