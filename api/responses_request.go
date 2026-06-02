// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"strings"
)

// This file ports the request-side OpenAI Responses API translation: turning a
// Responses input (a string, or an array of typed input items) plus optional
// instructions and a previous-response message chain into the internal message
// list the chat template consumes, and turning Responses flat tool definitions
// into the nested Chat Completions shape. Both are pure data transforms.
//
// Input items group across the array: consecutive function_call items
// accumulate into one synthesized assistant tool_calls message, reasoning items
// attach as reasoning_content to the assistant turn they precede, and developer
// or system items merge into a single leading system message.

// ResponsesInputItem is one entry of the Responses input array, parsed as an
// ordered object so optional fields can be probed for presence. The reference
// model normalizes a list/dict function_call_output to a JSON string before
// conversion; callers should do the same when building the value.
type ResponsesInputItem = jval

// tryParseJSON mirrors the reference helper: a string that, trimmed, begins with
// '{' or '[' and parses as JSON is returned as the parsed value; anything else
// passes through as a string value.
func tryParseJSON(s string) jval {
	t := strings.TrimSpace(s)
	if t == "" || (!strings.HasPrefix(t, "{") && !strings.HasPrefix(t, "[")) {
		return jstr(s)
	}
	if v, ok := parseOrdered(t); ok {
		return v
	}
	return jstr(s)
}

// flushPendingToolCalls drains accumulated tool calls into the message list. It
// merges into a trailing assistant message that has no tool_calls yet (avoiding
// duplicate assistant turns), otherwise it appends a new assistant message. Any
// pending reasoning is attached as reasoning_content. It returns the reasoning
// to carry forward: the input reasoning when nothing was flushed, or "" once
// reasoning has been consumed.
func flushPendingToolCalls(messages *[]jval, pending *[]jval, minMergeIndex int, pendingReasoning string) string {
	if len(*pending) == 0 {
		return pendingReasoning
	}
	calls := jval{kind: kindArray, arr: append([]jval{}, *pending...)}
	msgs := *messages
	if len(msgs) > 0 && len(msgs)-1 >= minMergeIndex &&
		msgs[len(msgs)-1].getString("role") == "assistant" &&
		!msgs[len(msgs)-1].hasField("tool_calls") {
		last := msgs[len(msgs)-1].setField("tool_calls", calls)
		if pendingReasoning != "" {
			last = last.setField("reasoning_content", jstr(pendingReasoning))
		}
		msgs[len(msgs)-1] = last
	} else {
		msg := jobj("role", jstr("assistant"), "tool_calls", calls)
		if pendingReasoning != "" {
			msg = msg.setField("reasoning_content", jstr(pendingReasoning))
		}
		*messages = append(msgs, msg)
	}
	*pending = (*pending)[:0]
	return ""
}

// consolidateSystemMessages moves every system message to the front, merged
// into one with blank-line separators, preserving the order of the rest.
func consolidateSystemMessages(messages []jval) []jval {
	var systemParts []string
	nonSystem := make([]jval, 0, len(messages))
	for _, msg := range messages {
		if msg.getString("role") == "system" {
			if c, ok := msg.getField("content"); ok && c.kind == kindString && c.s != "" {
				systemParts = append(systemParts, c.s)
			}
		} else {
			nonSystem = append(nonSystem, msg)
		}
	}
	if len(systemParts) == 0 {
		return messages
	}
	front := jobj("role", jstr("system"), "content", jstr(strings.Join(systemParts, "\n\n")))
	return append([]jval{front}, nonSystem...)
}

// effectiveItemType resolves an input item's type, defaulting an item that
// carries a role but no explicit type to "message" (the EasyInputMessage shape).
func effectiveItemType(item jval) string {
	if t, ok := item.getField("type"); ok && t.kind == kindString {
		return t.s
	}
	if item.hasField("role") {
		return "message"
	}
	return ""
}

// convertResponsesContent flattens a message item's content. A list of parts
// collapses to a newline-joined string unless an image part is present, in
// which case the converted parts list is kept so a VLM can extract images.
func convertResponsesContent(content jval) jval {
	if content.kind != kindArray {
		if content.kind == kindString {
			return content
		}
		return jstr("")
	}
	var textParts []string
	var converted []jval
	hasImage := false
	for _, part := range content.arr {
		switch part.kind {
		case kindObject:
			switch part.getString("type") {
			case "input_text", "text", "output_text":
				text := part.getString("text")
				textParts = append(textParts, text)
				converted = append(converted, jobj("type", jstr("text"), "text", jstr(text)))
			case "input_image":
				hasImage = true
				imageURL := part.getString("image_url")
				if !part.hasField("image_url") {
					imageURL = part.getString("url")
				}
				detail := "auto"
				if part.hasField("detail") {
					detail = part.getString("detail")
				}
				converted = append(converted, jobj(
					"type", jstr("input_image"),
					"image_url", jstr(imageURL),
					"detail", jstr(detail),
				))
			}
		case kindString:
			textParts = append(textParts, part.s)
			converted = append(converted, jobj("type", jstr("text"), "text", jstr(part.s)))
		}
	}
	if hasImage {
		return jval{kind: kindArray, arr: converted}
	}
	if len(textParts) > 0 {
		return jstr(strings.Join(textParts, "\n"))
	}
	return jstr("")
}

func contentOrEmpty(v jval) jval {
	switch v.kind {
	case kindString:
		return v
	case kindArray:
		if len(v.arr) > 0 {
			return v
		}
	}
	return jstr("")
}

// ConvertResponsesInputToMessages turns a Responses API input into the internal
// message list. instructions becomes a leading system message; previousMessages
// (from a previous_response_id chain) are prepended verbatim and never merged
// into by tool-call flushing.
func ConvertResponsesInputToMessages(input jval, instructions string, previousMessages []jval) []jval {
	var messages []jval
	var systemParts []string
	if instructions != "" {
		systemParts = append(systemParts, instructions)
	}

	if len(previousMessages) > 0 {
		messages = append(messages, previousMessages...)
	}
	currentMessageStart := len(messages)

	finish := func() []jval {
		if len(systemParts) > 0 {
			front := jobj("role", jstr("system"), "content", jstr(strings.Join(systemParts, "\n\n")))
			messages = append([]jval{front}, messages...)
		}
		return consolidateSystemMessages(messages)
	}

	switch input.kind {
	case kindString:
		messages = append(messages, jobj("role", jstr("user"), "content", input))
		return finish()
	case kindArray:
		// handled below
	default:
		return finish()
	}

	var pendingToolCalls []jval
	pendingReasoning := ""

	for _, item := range input.arr {
		if item.kind != kindObject {
			continue
		}
		itemType := effectiveItemType(item)

		switch itemType {
		case "message":
			pendingReasoning = flushPendingToolCalls(&messages, &pendingToolCalls, currentMessageStart, pendingReasoning)

			role := item.getString("role")
			if role == "" {
				role = "user"
			}
			if role == "developer" {
				role = "system"
			}

			content := jval{kind: kindNull}
			if c, ok := item.getField("content"); ok {
				content = convertResponsesContent(c)
			}
			content = contentOrEmpty(content)

			if role == "system" {
				sysText := ""
				if content.kind == kindString {
					sysText = content.s
				}
				systemParts = append(systemParts, sysText)
			} else {
				msg := jobj("role", jstr(role), "content", content)
				if role == "assistant" && pendingReasoning != "" {
					msg = msg.setField("reasoning_content", jstr(pendingReasoning))
					pendingReasoning = ""
				}
				messages = append(messages, msg)
			}

		case "reasoning":
			if summary, ok := item.getField("summary"); ok && summary.kind == kindArray && len(summary.arr) > 0 {
				var parts []string
				for _, s := range summary.arr {
					if s.kind == kindObject {
						if txt := s.getString("text"); txt != "" {
							parts = append(parts, txt)
						}
					}
				}
				pendingReasoning = strings.Join(parts, "\n")
			}

		case "function_call":
			callID := item.getString("call_id")
			if callID == "" {
				callID = item.getString("id")
			}
			if callID == "" {
				callID = newCallID()
			}
			args := "{}"
			if item.hasField("arguments") {
				if a := item.getString("arguments"); a != "" {
					args = a
				}
			}
			pendingToolCalls = append(pendingToolCalls, jobj(
				"id", jstr(callID),
				"type", jstr("function"),
				"function", jobj("name", jstr(item.getString("name")), "arguments", tryParseJSON(args)),
			))

		case "function_call_output":
			pendingReasoning = flushPendingToolCalls(&messages, &pendingToolCalls, currentMessageStart, pendingReasoning)
			messages = append(messages, jobj(
				"role", jstr("tool"),
				"tool_call_id", jstr(item.getString("call_id")),
				"content", jstr(item.getString("output")),
			))
		}
	}

	flushPendingToolCalls(&messages, &pendingToolCalls, currentMessageStart, pendingReasoning)
	return finish()
}

// ConvertResponsesTools maps Responses flat tool definitions to the nested Chat
// Completions shape, skipping non-function tools the local model cannot run. It
// returns nil when no function tools remain.
func ConvertResponsesTools(tools []jval) []Tool {
	if len(tools) == 0 {
		return nil
	}
	var result []Tool
	for _, tool := range tools {
		if tool.getString("type") != "function" {
			continue
		}
		name := tool.getString("name")
		if name == "" {
			continue
		}
		fn := FunctionDef{Name: name}
		if d := tool.getString("description"); d != "" {
			fn.Description = d
		}
		if p, ok := tool.getField("parameters"); ok && (p.kind == kindObject || p.kind == kindArray) {
			fn.Parameters = []byte(p.dumpCompact())
		}
		if s, ok := tool.getField("strict"); ok && s.kind == kindBool {
			b := s.s == "true"
			fn.Strict = &b
		}
		result = append(result, Tool{Type: "function", Function: fn})
	}
	if len(result) == 0 {
		return nil
	}
	return result
}
