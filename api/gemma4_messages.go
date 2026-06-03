// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "github.com/tamnd/fastmlx/adapter"

// ExtractGemma4Messages ports extract_gemma4_messages: it converts OpenAI-format
// messages into the shape the Gemma 4 chat template expects. The template has no
// role="tool" handling, so tool results are folded onto a model-role turn as a
// tool_responses list, with each entry's function name resolved from the
// preceding assistant turn's tool_calls by tool_call_id (falling back to the raw
// id, then "unknown"). Assistant tool_calls are preserved with their arguments
// JSON-parsed, prior thought blocks are stripped from assistant content per
// Gemma 4's multi-turn rule, and user/system turns keep image_url parts so the
// VLM path can see them.
//
// The reference's optional tool-result truncation runs only when both a token
// budget and a tokenizer are supplied; that is the tokenizer seam, so this
// GPU-free port omits it (a caller that needs truncation pre-truncates the
// content). Everything else, including the shared consolidate/drop-void/merge
// cleanup tail, is the pure transform.
func ExtractGemma4Messages(messages []jval) []jval {
	var processed []jval

	i := 0
	for i < len(messages) {
		msg := messages[i]
		role := msg.getString("role")
		if role == "" {
			role = "user"
		}
		if role == "developer" {
			role = "system"
		}

		if role == "tool" {
			// Orphaned tool result with no preceding assistant turn: attach it
			// to a synthetic assistant turn that carries no content.
			toolCallID := msg.getString("tool_call_id")
			content := gemma4Content(msg)
			response := gemma4TryParse(content)
			name := toolCallID
			if name == "" {
				name = "unknown"
			}
			processed = append(processed, jobj(
				"role", jstr("assistant"),
				"content", jstr(""),
				"tool_responses", jval{kind: kindArray, arr: []jval{
					jobj("name", jstr(name), "response", response),
				}},
				preserveRoleBoundaryKey, jval{kind: kindBool, s: "true"},
			))
			i++
			continue
		}

		if role == "assistant" {
			toolCalls := gemma4ToolCalls(msg)

			// tool_call_id -> function name lookup for this turn's calls.
			idToName := map[string]string{}
			for _, tc := range toolCalls {
				if tc.kind != kindObject {
					continue
				}
				if tcID := tc.getString("id"); tcID != "" {
					fn := tc.getOr("function", jval{kind: kindObject})
					idToName[tcID] = fn.getString("name")
				}
			}

			content := gemma4Content(msg)
			if content.kind == kindString {
				content = jstr(adapter.StripLeadingThoughts(content.s))
			}

			outMsg := jobj("role", jstr("assistant"), "content", gemma4OrEmpty(content))

			if len(toolCalls) > 0 {
				outCalls := make([]jval, 0, len(toolCalls))
				for _, tc := range toolCalls {
					fn := tc.getOr("function", jval{kind: kindObject})
					args := fn.getOr("arguments", jstr("{}"))
					outCalls = append(outCalls, jobj(
						"id", jstr(tc.getString("id")),
						"function", jobj(
							"name", jstr(fn.getString("name")),
							"arguments", gemma4TryParse(args),
						),
					))
				}
				outMsg = outMsg.setField("tool_calls", jval{kind: kindArray, arr: outCalls})
				outMsg = outMsg.setField(preserveRoleBoundaryKey, jval{kind: kindBool, s: "true"})
			}

			i++

			// Fold any immediately following tool results into one model turn.
			var toolResponses []jval
			for i < len(messages) && messages[i].getString("role") == "tool" {
				tr := messages[i]
				tcID := tr.getString("tool_call_id")
				response := gemma4TryParse(gemma4Content(tr))
				name := idToName[tcID]
				if name == "" {
					name = tcID
				}
				if name == "" {
					name = "unknown"
				}
				toolResponses = append(toolResponses, jobj("name", jstr(name), "response", response))
				i++
			}
			if len(toolResponses) > 0 {
				outMsg = outMsg.setField("tool_responses", jval{kind: kindArray, arr: toolResponses})
			}
			processed = append(processed, outMsg)
			continue
		}

		// user / system: keep image_url parts so the VLM path can see them.
		content := msg.getOr("content", jstr(""))
		if content.kind == kindArray {
			multimodal := ExtractMultimodalContentList(content)
			hasImages := false
			for _, p := range multimodal.arr {
				if p.getString("type") == "image_url" {
					hasImages = true
					break
				}
			}
			if hasImages {
				content = multimodal
			} else {
				content = jstr(ExtractTextFromContentList(content))
			}
		}
		// Reference: content if content is not None else "" (only null maps to "").
		if content.kind == kindNull {
			content = jstr("")
		}
		processed = append(processed, jobj("role", jstr(role), "content", content))
		i++
	}

	return mergeConsecutiveRoles(DropVoidAssistantMessages(consolidateSystemMessages(processed)))
}

// gemma4Content resolves a message's content field for the tool/assistant paths:
// a missing field is the empty string, and a content array is flattened to its
// text the way the reference _extract_text_from_content_list does.
func gemma4Content(msg jval) jval {
	content, ok := msg.getField("content")
	if !ok {
		return jstr("")
	}
	if content.kind == kindArray {
		return jstr(ExtractTextFromContentList(content))
	}
	return content
}

// gemma4ToolCalls returns a message's tool_calls as a slice, treating a missing
// or non-array value as empty (the reference `msg.get("tool_calls") or []`).
func gemma4ToolCalls(msg jval) []jval {
	tc, ok := msg.getField("tool_calls")
	if !ok || tc.kind != kindArray {
		return nil
	}
	return tc.arr
}

// gemma4TryParse mirrors the reference _try_parse_json call sites: a string is
// parsed as JSON when it can be, and any non-string value passes through.
func gemma4TryParse(v jval) jval {
	if v.kind == kindString {
		return TryParseJSON(v.s)
	}
	return v
}

// gemma4OrEmpty mirrors the reference `content or ""`: a truthy value (a
// non-empty string, list, or object, a non-zero number, or true) stays, and any
// falsy value becomes "".
func gemma4OrEmpty(v jval) jval {
	if pythonTruthy(v) {
		return v
	}
	return jstr("")
}
