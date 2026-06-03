// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "strings"

// This file ports the Harmony (gpt-oss) request-side variant of the Anthropic
// translation, convert_anthropic_to_internal_harmony. Unlike the general
// convertAnthropicToInternal it always preserves tool structure for the Harmony
// chat template: tool_use blocks become an assistant tool_calls field and each
// tool_result becomes a separate role="tool" message, both with their arguments
// and content kept as parsed JSON objects rather than re-stringified. It takes
// no native-tool/native-reasoning/preserve-image options because the Harmony
// path makes none of those choices: images and thinking blocks are dropped, and
// tool structure is always preserved. The token-budget truncation of tool
// results is the one tokenizer seam and is omitted here (the caller
// pre-truncates), matching how convertAnthropicToInternal omits it.

// harmonyToolResult holds a tool_result block's id and its resolved content
// while a user turn is being processed, before it is emitted as a tool message.
type harmonyToolResult struct {
	id      string
	content jval
}

// ConvertAnthropicToInternalHarmony turns an Anthropic request into the internal
// message list for Harmony models. System entries are normalized exactly as the
// general path does; consecutive same-role messages are merged at the end (the
// Harmony path sets no preserve-boundary markers, so an assistant turn that
// issued tool calls can still merge with an adjacent assistant turn, matching
// the reference).
func ConvertAnthropicToInternalHarmony(system jval, messages []AnthropicInMessage) []jval {
	var processed []jval

	systemText, normalized := normalizeInMessagesSystem(messages, system)
	if systemText != "" {
		processed = append(processed, jobj("role", jstr("system"), "content", jstr(systemText)))
	}

	for _, msg := range normalized {
		role := msg.Role
		content := msg.Content

		switch content.kind {
		case kindString:
			processed = append(processed, jobj("role", jstr(role), "content", jstr(content.s)))
			continue
		case kindArray:
			// handled below
		default:
			processed = append(processed, jobj("role", jstr(role), "content", jstr(stringifyUnknownContent(content))))
			continue
		}

		var textParts []string
		var toolCalls []jval
		var toolResults []harmonyToolResult

		for _, block := range content.arr {
			if block.kind != kindObject {
				continue
			}
			switch block.getString("type") {
			case "text":
				textParts = append(textParts, block.getString("text"))
			case "tool_use":
				toolCalls = append(toolCalls, toolCallObject(block))
			case "tool_result":
				rc, present := block.getField("content")
				toolResults = append(toolResults, harmonyToolResult{
					id:      block.getString("tool_use_id"),
					content: harmonyToolResultContent(rc, present),
				})
			case "thinking":
				continue
			case "document":
				textParts = append(textParts, decodeDocumentBlock(block))
			}
		}

		switch role {
		case "assistant":
			msgDict := jobj("role", jstr("assistant"), "content", jstr(strings.Join(textParts, "\n")))
			if len(toolCalls) > 0 {
				msgDict = msgDict.setField("tool_calls", jval{kind: kindArray, arr: toolCalls})
			}
			processed = append(processed, msgDict)
		case "user":
			if len(textParts) > 0 {
				processed = append(processed, jobj("role", jstr("user"), "content", jstr(strings.Join(textParts, "\n"))))
			}
			for _, tr := range toolResults {
				processed = append(processed, jobj(
					"role", jstr("tool"),
					"tool_call_id", jstr(tr.id),
					"content", tr.content,
				))
			}
		default:
			processed = append(processed, jobj("role", jstr(role), "content", jstr(strings.Join(textParts, "\n"))))
		}
	}

	return mergeConsecutiveRoles(processed)
}

// harmonyToolResultContent resolves a tool_result's content the way the Harmony
// path does so the chat template's tojson filter receives parsed JSON, not a
// re-encoded string. A missing content defaults to the empty string. A string
// that parses as JSON and is not the literal null is passed through as the
// parsed value (the reference guards the string branch with "parsed is not
// None", so a content of "null" stays the string "null"); otherwise the string
// is kept. A content list is flattened to text and then parsed if that text is
// itself JSON (the list branch has no null guard, so a list joining to "null"
// becomes null, matching the reference). Any other shape (an object, number,
// bool, or explicit null) passes through unchanged.
func harmonyToolResultContent(content jval, present bool) jval {
	if !present {
		return jstr("")
	}
	switch content.kind {
	case kindString:
		if parsed, ok := parseOrdered(content.s); ok && parsed.kind != kindNull {
			return parsed
		}
		return content
	case kindArray:
		extracted := extractToolResultContent(content)
		if parsed, ok := parseOrdered(extracted); ok {
			return parsed
		}
		return jstr(extracted)
	default:
		return content
	}
}
