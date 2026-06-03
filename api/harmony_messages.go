// SPDX-License-Identifier: MIT OR Apache-2.0

package api

// ExtractHarmonyMessages ports extract_harmony_messages: it prepares messages
// for Harmony (gpt-oss) models, whose chat template expects standard OpenAI
// format and converts role="tool" turns and assistant tool_calls itself. Unlike
// ExtractGemma4Messages it preserves the tool role and the tool_calls field
// intact; its one transform is to JSON-parse content and tool-call arguments
// that arrive as JSON strings, because the template applies a |tojson filter and
// would otherwise double-encode an already-encoded string.
//
// The reference's optional tool-result truncation runs only when both a token
// budget and a tokenizer are supplied (parsing and pretty-printing the result so
// the cut lands on a line boundary, then wrapping a broken-JSON cut back into a
// dict); that is the tokenizer seam, so this GPU-free port omits it (a caller
// that needs truncation pre-truncates) and keeps the no-truncation path, which
// simply JSON-parses the tool content. The shared consolidate → drop-void →
// merge cleanup tail is ported as-is.
func ExtractHarmonyMessages(messages []jval) []jval {
	var processed []jval

	for _, msg := range messages {
		role := msg.getString("role")
		if role == "" {
			role = "user"
		}
		content, hasContent := msg.getField("content")
		if role == "developer" {
			role = "system"
		}

		if role == "tool" {
			// list content flattens to text; any other falsy content becomes "".
			var toolContent jval
			if hasContent && content.kind == kindArray {
				toolContent = jstr(ExtractTextFromContentList(content))
			} else if hasContent && pythonTruthy(content) {
				toolContent = content
			} else {
				toolContent = jstr("")
			}
			processed = append(processed, jobj(
				"role", jstr("tool"),
				"tool_call_id", jstr(msg.getString("tool_call_id")),
				"content", gemma4TryParse(toolContent),
			))
			continue
		}

		if role == "assistant" {
			msgDict := jobj("role", jstr(role), "content", harmonyContentString(content, hasContent))
			toolCalls := gemma4ToolCalls(msg)
			if len(toolCalls) > 0 {
				outCalls := make([]jval, 0, len(toolCalls))
				for _, tc := range toolCalls {
					if tc.kind != kindObject {
						continue
					}
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
				msgDict = msgDict.setField("tool_calls", jval{kind: kindArray, arr: outCalls})
				msgDict = msgDict.setField(preserveRoleBoundaryKey, jval{kind: kindBool, s: "true"})
			}
			processed = append(processed, msgDict)
			continue
		}

		// user / system / other
		processed = append(processed, jobj("role", jstr(role), "content", harmonyContentString(content, hasContent)))
	}

	return mergeConsecutiveRoles(DropVoidAssistantMessages(consolidateSystemMessages(processed)))
}

// harmonyContentString renders a message's content the way the reference does
// for assistant and plain turns: a missing or null content is "", a string
// passes through, a content array is flattened to its text, and any other value
// takes the Python str() fallback. The API only sends string, list, or null
// content; the residual covers a bool (Python "True"/"False") and a number (its
// literal), with object/array-of-non-content shapes (never emitted) rendered as
// their JSON form rather than a Python repr.
func harmonyContentString(content jval, present bool) jval {
	if !present || content.kind == kindNull {
		return jstr("")
	}
	switch content.kind {
	case kindString:
		return content
	case kindArray:
		return jstr(ExtractTextFromContentList(content))
	case kindBool:
		if content.s == "true" {
			return jstr("True")
		}
		return jstr("False")
	case kindNumber:
		return jstr(canonicalNumberLiteral(content.s))
	default:
		return jstr(content.dump())
	}
}
