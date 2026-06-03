// SPDX-License-Identifier: MIT OR Apache-2.0

package api

// ExtractTextContent ports extract_text_content: the message preprocessor for
// the OpenAI-format /v1/chat/completions route, the sibling of
// ExtractGemma4Messages and ExtractHarmonyMessages for models whose chat
// template expects plain string content. It flattens content arrays to text,
// reconstructs historical reasoning, renames developer to system, and folds tool
// turns and assistant tool_calls into the shape the template understands.
//
// Two model-bound decisions are injected as plain bools. nativeToolCalling
// stands in for the reference's _chat_template_supports_tool_role(tokenizer):
// when true the tool role and structured tool_calls are preserved (the template
// renders them), when false a tool result is rewritten as a bracketed user turn
// and an assistant's calls are appended to its content as bracketed markup.
// nativeReasoningContent selects how a stored reasoning trace is replayed (a
// separate reasoning_content field versus an inline <think> block), exactly as
// applyReasoningReconstruction does.
//
// The reference's optional tool-result truncation runs only with both a token
// budget and a tokenizer; that is the tokenizer seam, so this GPU-free port omits
// it (a caller needing truncation pre-truncates) and keeps the no-truncation
// path. The shared consolidate -> drop-void -> merge cleanup tail is ported
// as-is.
func ExtractTextContent(messages []jval, nativeToolCalling, nativeReasoningContent bool) []jval {
	var processed []jval

	for _, msg := range messages {
		role := msg.getString("role")
		content, hasContent := msg.getField("content")
		if !hasContent {
			content = jnull()
		}
		reasoning := msg.getString("reasoning_content")
		newContent, reasoningOut, hasReasoning := applyReasoningReconstruction(role, content, reasoning, nativeReasoningContent)

		if role == "developer" {
			role = "system"
		}

		if role == "tool" {
			toolCallID := msg.getString("tool_call_id")
			var toolContent jval
			if newContent.kind == kindArray {
				toolContent = jstr(ExtractTextFromContentList(newContent))
			} else if pythonTruthy(newContent) {
				toolContent = newContent
			} else {
				toolContent = jstr("")
			}
			if nativeToolCalling {
				processed = append(processed, jobj(
					"role", jstr("tool"),
					"tool_call_id", jstr(toolCallID),
					"content", toolContent,
				))
			} else {
				text := "[Tool Result (" + toolCallID + ")]: " + pythonStrValue(toolContent)
				processed = append(processed, jobj(
					"role", jstr("user"),
					"content", jstr(text),
					preserveRoleBoundaryKey, jbool(true),
				))
			}
			continue
		}

		toolCalls := gemma4ToolCalls(msg)
		if role == "assistant" && len(toolCalls) > 0 {
			c := newContent
			if c.kind == kindArray {
				c = jstr(ExtractTextFromContentList(c))
			}
			contentField := c
			if !pythonTruthy(c) {
				contentField = jstr("")
			}
			msgDict := jobj("role", jstr(role), "content", contentField)
			if hasReasoning {
				msgDict = msgDict.setField("reasoning_content", jstr(reasoningOut))
			}
			if name, ok := msg.getField("name"); ok && pythonTruthy(name) {
				msgDict = msgDict.setField("name", name)
			}
			if partial, ok := msg.getField("partial"); ok && pythonTruthy(partial) {
				msgDict = msgDict.setField("partial", jbool(true))
			}

			if nativeToolCalling {
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
			} else {
				var calls []string
				for _, tc := range toolCalls {
					if tc.kind != kindObject {
						continue
					}
					fn := tc.getOr("function", jval{kind: kindObject})
					name := "unknown"
					if nv, ok := fn.getField("name"); ok {
						name = pythonStrValue(nv)
					}
					args := "{}"
					if av, ok := fn.getField("arguments"); ok {
						args = pythonStrValue(av)
					}
					calls = append(calls, "[Calling tool: "+name+"("+args+")]")
				}
				text := contentField.s
				if len(calls) > 0 {
					prefix := ""
					if text != "" {
						prefix = text + "\n"
					}
					text = prefix + joinLines(calls)
				}
				msgDict = msgDict.setField("content", jstr(text))
			}

			msgDict = msgDict.setField(preserveRoleBoundaryKey, jbool(true))
			processed = append(processed, msgDict)
			continue
		}

		// Regular user / system / assistant-without-calls message. The extra
		// fields are appended in the reference's order: name, partial,
		// reasoning_content.
		var contentField jval
		switch {
		case newContent.kind == kindNull:
			contentField = jstr("")
		case newContent.kind == kindString:
			contentField = newContent
		case newContent.kind == kindArray:
			contentField = jstr(ExtractTextFromContentList(newContent))
		default:
			contentField = jstr(pythonStrValue(newContent))
		}
		msgDict := jobj("role", jstr(role), "content", contentField)
		if name, ok := msg.getField("name"); ok && pythonTruthy(name) {
			msgDict = msgDict.setField("name", name)
		}
		if partial, ok := msg.getField("partial"); ok && pythonTruthy(partial) {
			msgDict = msgDict.setField("partial", jbool(true))
		}
		if hasReasoning {
			msgDict = msgDict.setField("reasoning_content", jstr(reasoningOut))
		}
		processed = append(processed, msgDict)
	}

	return mergeConsecutiveRoles(DropVoidAssistantMessages(consolidateSystemMessages(processed)))
}

// pythonStrValue renders a jval the way a Python f-string str() would for the
// values that reach the tool-result and tool-call fallback paths: a string is
// itself, a number keeps its literal, a bool is "True"/"False", and null is
// "None". An object or array (never emitted into these fields by the API) falls
// back to its JSON form rather than a Python repr.
func pythonStrValue(v jval) string {
	switch v.kind {
	case kindString:
		return v.s
	case kindNumber:
		return canonicalNumberLiteral(v.s)
	case kindBool:
		if v.s == "true" {
			return "True"
		}
		return "False"
	case kindNull:
		return "None"
	default:
		return v.dump()
	}
}

// joinLines joins with "\n", matching Python "\n".join(parts).
func joinLines(parts []string) string {
	out := ""
	for i, p := range parts {
		if i > 0 {
			out += "\n"
		}
		out += p
	}
	return out
}
