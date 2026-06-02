// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/base64"
	"strings"
)

// This file ports the request-side Anthropic translation: turning a Messages
// API request into the internal message list the chat-template layer consumes.
// It is a pure data transform. Two inputs are decided upstream from the
// tokenizer and passed in as flags here: whether the active chat template
// renders tool messages natively, and whether it reads a top-level
// reasoning_content field. Token-budget truncation of tool results also needs a
// tokenizer, so it is left as a follow-up; without one, tool results pass
// through whole, exactly as the reference does when no tokenizer is supplied.
//
// Anthropic document blocks are the one place the reference embeds a
// project-named, em-dashed placeholder into the prompt. This port rewords those
// placeholders cleanly; they are covered by direct unit tests rather than
// reference-captured fixtures, the same intentional-divergence class as the
// thinking-block signature.

const billingHeaderPrefix = "x-anthropic-billing-header:"

// preserveRoleBoundaryKey marks a message that must not be merged into an
// adjacent same-role message (an assistant turn that issued tool calls, or a
// user turn that carried tool results). It travels as a plain key the chat
// template ignores, matching the reference sentinel.
const preserveRoleBoundaryKey = "_preserve_role_boundary"

// AnthropicInMessage is one entry of the request's messages array. Content is
// the raw JSON value, either a string or an array of content blocks.
type AnthropicInMessage struct {
	Role    string
	Content jval
}

// AnthropicConvertOptions carries the decisions made upstream from the
// tokenizer. NativeToolCalling is the reference's tokenizer-derived
// native_tool_calling; the other two mirror the like-named parameters.
type AnthropicConvertOptions struct {
	NativeToolCalling      bool
	PreserveImages         bool
	NativeReasoningContent bool
}

func jbool(b bool) jval {
	if b {
		return jval{kind: kindBool, s: "true"}
	}
	return jval{kind: kindBool, s: "false"}
}

// setField returns a copy of object v with key set to val, replacing an
// existing key in place or appending a new one at the end.
func (v jval) setField(key string, val jval) jval {
	out := jval{kind: kindObject, obj: make([]jkv, len(v.obj))}
	copy(out.obj, v.obj)
	for i := range out.obj {
		if out.obj[i].k == key {
			out.obj[i].v = val
			return out
		}
	}
	out.obj = append(out.obj, jkv{key, val})
	return out
}

func (v jval) hasField(key string) bool {
	_, ok := v.getField(key)
	return ok
}

// decodeDocumentBlock renders an Anthropic document content block as text. A
// text/plain document is base64-decoded inline; any other media type yields a
// short placeholder, since this server does not parse PDFs or other binary
// documents.
func decodeDocumentBlock(block jval) string {
	source, _ := block.getField("source")
	mediaType := source.getString("media_type")
	data := source.getString("data")
	title := block.getString("title")

	if mediaType == "text/plain" && data != "" {
		decoded, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			label := title
			if label == "" {
				label = "untitled"
			}
			return "[Document: " + label + ": failed to decode]"
		}
		label := ""
		if title != "" {
			label = "[Document: " + title + "]\n"
		}
		return label + string(decoded)
	}

	label := title
	if label == "" {
		label = "untitled"
	}
	return "[Document: " + label + " (" + mediaType + "): document parsing is not available, send as text instead.]"
}

// appendImagePart converts an Anthropic image block to an OpenAI-style
// image_url part and appends it.
func appendImagePart(parts []jval, block jval) []jval {
	source, _ := block.getField("source")
	switch source.getString("type") {
	case "base64":
		mediaType := source.getString("media_type")
		if mediaType == "" {
			mediaType = "image/jpeg"
		}
		url := "data:" + mediaType + ";base64," + source.getString("data")
		return append(parts, jobj("type", jstr("image_url"), "image_url", jobj("url", jstr(url))))
	case "url":
		return append(parts, jobj("type", jstr("image_url"), "image_url", jobj("url", jstr(source.getString("url")))))
	}
	return parts
}

// extractImagesFromToolResult pulls image blocks out of a tool_result's content
// so a VLM can still see them.
func extractImagesFromToolResult(content jval, parts []jval) []jval {
	switch content.kind {
	case kindArray:
		for _, item := range content.arr {
			if item.kind == kindObject && item.getString("type") == "image" {
				parts = appendImagePart(parts, item)
			}
		}
	case kindObject:
		if content.getString("type") == "image" {
			parts = appendImagePart(parts, content)
		}
	}
	return parts
}

// extractToolResultContent flattens a tool_result's content to text. A bare
// string passes through; a list joins its text items with newlines; a non-text
// object is dumped as JSON (json.dumps default, ensure_ascii=True).
func extractToolResultContent(content jval) string {
	switch content.kind {
	case kindString:
		return content.s
	case kindArray:
		var textParts []string
		for _, item := range content.arr {
			switch item.kind {
			case kindObject:
				if item.getString("type") == "text" {
					textParts = append(textParts, item.getString("text"))
				}
			case kindString:
				textParts = append(textParts, item.s)
			}
		}
		return strings.Join(textParts, "\n")
	case kindObject:
		if content.getString("type") == "text" {
			return content.getString("text")
		}
		return content.dumpASCII()
	case kindNumber:
		return content.s
	case kindBool:
		if content.s == "true" {
			return "True"
		}
		return "False"
	case kindNull:
		return "None"
	}
	return ""
}

// extractSystemText flattens the request system field to text, joining text
// blocks with newlines and skipping billing-header blocks (random values that
// would poison the prefix cache).
func extractSystemText(system jval) string {
	switch system.kind {
	case kindString:
		return system.s
	case kindArray:
		var parts []string
		for _, block := range system.arr {
			if block.kind != kindObject || block.getString("type") != "text" {
				continue
			}
			text := block.getString("text")
			if strings.HasPrefix(text, billingHeaderPrefix) {
				continue
			}
			parts = append(parts, text)
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

// normalizeInMessagesSystem lifts any role="system" entries out of the messages
// array (some clients send system content inline), merges them with the
// canonical system field, and returns the combined system text plus the
// messages with system entries removed.
func normalizeInMessagesSystem(messages []AnthropicInMessage, system jval) (string, []AnthropicInMessage) {
	var extracted []string
	filtered := make([]AnthropicInMessage, 0, len(messages))
	for _, msg := range messages {
		if msg.Role != "system" {
			filtered = append(filtered, msg)
			continue
		}
		switch msg.Content.kind {
		case kindString:
			if msg.Content.s != "" {
				extracted = append(extracted, msg.Content.s)
			}
		case kindArray:
			for _, block := range msg.Content.arr {
				if block.kind == kindObject && block.getString("type") == "text" {
					if text := block.getString("text"); text != "" {
						extracted = append(extracted, text)
					}
				}
			}
		}
	}

	base := ""
	if system.kind == kindString || system.kind == kindArray {
		base = extractSystemText(system)
	}
	if len(extracted) == 0 {
		return base, filtered
	}
	extra := strings.Join(extracted, "\n")
	var combined []string
	for _, p := range []string{base, extra} {
		if p != "" {
			combined = append(combined, p)
		}
	}
	return strings.Join(combined, "\n\n"), filtered
}

// buildMessageFromParts assembles one internal message from accumulated text
// and image parts, or reports false when there is nothing to emit.
func buildMessageFromParts(role string, textParts []string, imageParts []jval) (jval, bool) {
	if len(imageParts) > 0 {
		parts := append([]jval{}, imageParts...)
		if len(textParts) > 0 {
			parts = append(parts, jobj("type", jstr("text"), "text", jstr(strings.Join(textParts, "\n"))))
		}
		return jobj("role", jstr(role), "content", jval{kind: kindArray, arr: parts}), true
	}
	if len(textParts) > 0 {
		return jobj("role", jstr(role), "content", jstr(strings.Join(textParts, "\n"))), true
	}
	return jval{}, false
}

// toolCallObject builds an internal tool_calls entry whose arguments is the
// parsed input object (a string input is re-parsed when it is valid JSON).
func toolCallObject(block jval) jval {
	id := block.getString("id")
	if id == "" {
		id = newCallID()
	}
	input, ok := block.getField("input")
	if !ok {
		input = jval{kind: kindObject}
	} else if input.kind == kindString {
		if parsed, parsedOK := parseOrdered(input.s); parsedOK {
			input = parsed
		}
	}
	return jobj(
		"id", jstr(id),
		"function", jobj("name", jstr(block.getString("name")), "arguments", input),
	)
}

// convertAnthropicToInternal turns an Anthropic request into the internal
// message list. Consecutive same-role messages are merged at the end for chat
// templates that require strict alternation.
func convertAnthropicToInternal(system jval, messages []AnthropicInMessage, opts AnthropicConvertOptions) []jval {
	var out []jval

	systemText, normalized := normalizeInMessagesSystem(messages, system)
	if systemText != "" {
		out = append(out, jobj("role", jstr("system"), "content", jstr(systemText)))
	}

	for _, msg := range normalized {
		role := msg.Role
		content := msg.Content

		switch content.kind {
		case kindString:
			out = append(out, jobj("role", jstr(role), "content", jstr(content.s)))
			continue
		case kindArray:
			// handled below
		default:
			out = append(out, jobj("role", jstr(role), "content", jstr(stringifyUnknownContent(content))))
			continue
		}

		if opts.NativeToolCalling && role == "assistant" {
			out = append(out, convertNativeAssistant(role, content, opts)...)
			continue
		}
		if opts.NativeToolCalling && role == "user" {
			out = append(out, convertNativeUser(role, content, opts)...)
			continue
		}

		out = append(out, convertFallbackBlocks(role, content, opts))
	}

	return mergeConsecutiveRoles(out)
}

// convertNativeAssistant handles an assistant message whose template renders
// tool calls natively: text and thinking collapse into content (or
// reasoning_content), tool_use blocks become tool_calls.
func convertNativeAssistant(role string, content jval, opts AnthropicConvertOptions) []jval {
	var textParts, thinkingParts []string
	var imageParts, toolCalls []jval

	for _, block := range content.arr {
		if block.kind != kindObject {
			continue
		}
		switch block.getString("type") {
		case "text":
			textParts = append(textParts, block.getString("text"))
		case "image":
			if opts.PreserveImages {
				imageParts = appendImagePart(imageParts, block)
			}
		case "tool_use":
			toolCalls = append(toolCalls, toolCallObject(block))
		case "thinking":
			if t := block.getString("thinking"); t != "" {
				if opts.NativeReasoningContent {
					thinkingParts = append(thinkingParts, t)
				} else {
					textParts = append(textParts, "<think>\n"+t+"\n</think>")
				}
			}
		case "document":
			textParts = append(textParts, decodeDocumentBlock(block))
		}
	}

	msg, ok := buildMessageFromParts(role, textParts, imageParts)
	if !ok {
		msg = jobj("role", jstr(role), "content", jstr(""))
	}
	if len(thinkingParts) > 0 {
		msg = msg.setField("reasoning_content", jstr(strings.Join(thinkingParts, "\n")))
	}
	if len(toolCalls) > 0 {
		msg = msg.setField("tool_calls", jval{kind: kindArray, arr: toolCalls})
		msg = msg.setField(preserveRoleBoundaryKey, jbool(true))
	}
	return []jval{msg}
}

// convertNativeUser handles a user message whose template renders tool results
// natively: each tool_result becomes a separate role="tool" message, flushing
// any text or images accumulated before it.
func convertNativeUser(role string, content jval, opts AnthropicConvertOptions) []jval {
	var out []jval
	var textParts []string
	var imageParts []jval
	sawToolResult := false

	for _, block := range content.arr {
		if block.kind != kindObject {
			continue
		}
		switch block.getString("type") {
		case "text":
			textParts = append(textParts, block.getString("text"))
		case "image":
			if opts.PreserveImages {
				imageParts = appendImagePart(imageParts, block)
			}
		case "tool_result":
			if msg, ok := buildMessageFromParts(role, textParts, imageParts); ok {
				out = append(out, msg)
			}
			textParts = nil
			imageParts = nil
			sawToolResult = true
			resultContent, _ := block.getField("content")
			out = append(out, jobj(
				"role", jstr("tool"),
				"tool_call_id", jstr(block.getString("tool_use_id")),
				"content", jstr(extractToolResultContent(resultContent)),
			))
			if opts.PreserveImages {
				imageParts = extractImagesFromToolResult(resultContent, imageParts)
			}
		case "thinking":
			if t := block.getString("thinking"); t != "" && !opts.NativeReasoningContent {
				textParts = append(textParts, "<think>\n"+t+"\n</think>")
			}
		case "document":
			textParts = append(textParts, decodeDocumentBlock(block))
		}
	}

	if msg, ok := buildMessageFromParts(role, textParts, imageParts); ok {
		out = append(out, msg)
	} else if !sawToolResult {
		out = append(out, jobj("role", jstr(role), "content", jstr("")))
	}
	return out
}

// convertFallbackBlocks handles the non-native path: tool uses and results are
// inlined as bracket markup, thinking is inlined as <think> unless the template
// reads a native reasoning field.
func convertFallbackBlocks(role string, content jval, opts AnthropicConvertOptions) jval {
	var textParts, thinkingParts []string
	var imageParts []jval
	sawToolMarkup := false

	for _, block := range content.arr {
		if block.kind != kindObject {
			continue
		}
		switch block.getString("type") {
		case "text":
			textParts = append(textParts, block.getString("text"))
		case "image":
			if opts.PreserveImages {
				imageParts = appendImagePart(imageParts, block)
			}
		case "tool_use":
			input, _ := block.getField("input")
			if !block.hasField("input") {
				input = jval{kind: kindObject}
			}
			textParts = append(textParts, "[Calling tool: "+block.getString("name")+"("+input.dumpASCII()+")]")
			sawToolMarkup = true
		case "tool_result":
			resultContent, _ := block.getField("content")
			result := extractToolResultContent(resultContent)
			prefix := "[Tool Result"
			if f, ok := block.getField("is_error"); ok && f.kind == kindBool && f.s == "true" {
				prefix = "[Tool Error"
			}
			textParts = append(textParts, prefix+" ("+block.getString("tool_use_id")+")]: "+result)
			sawToolMarkup = true
			if opts.PreserveImages {
				imageParts = extractImagesFromToolResult(resultContent, imageParts)
			}
		case "thinking":
			if t := block.getString("thinking"); t != "" {
				if opts.NativeReasoningContent && role == "assistant" {
					thinkingParts = append(thinkingParts, t)
				} else if !opts.NativeReasoningContent {
					textParts = append(textParts, "<think>\n"+t+"\n</think>")
				}
			}
		case "document":
			textParts = append(textParts, decodeDocumentBlock(block))
		}
	}

	msg, ok := buildMessageFromParts(role, textParts, imageParts)
	if !ok {
		msg = jobj("role", jstr(role), "content", jstr(""))
	}
	if len(thinkingParts) > 0 {
		msg = msg.setField("reasoning_content", jstr(strings.Join(thinkingParts, "\n")))
	}
	if sawToolMarkup {
		msg = msg.setField(preserveRoleBoundaryKey, jbool(true))
	}
	return msg
}

func stringifyUnknownContent(v jval) string {
	switch v.kind {
	case kindNull:
		return "None"
	case kindBool:
		if v.s == "true" {
			return "True"
		}
		return "False"
	case kindNumber, kindString:
		return v.s
	}
	return v.dumpASCII()
}

var mergeableRoles = map[string]bool{"user": true, "assistant": true}

// mergeConsecutiveRoles merges adjacent messages that share a mergeable role,
// unless either carries the preserve-boundary marker. String contents join with
// a blank line; when either side is a parts list, both become lists and concat.
func mergeConsecutiveRoles(messages []jval) []jval {
	if len(messages) == 0 {
		return messages
	}
	merged := []jval{messages[0]}
	for _, msg := range messages[1:] {
		prev := merged[len(merged)-1]
		prevRole := prev.getString("role")
		if msg.getString("role") == prevRole && mergeableRoles[prevRole] &&
			!boundaryMarked(prev) && !boundaryMarked(msg) {
			prevContent, _ := prev.getField("content")
			newContent, _ := msg.getField("content")
			merged[len(merged)-1] = prev.setField("content", mergeContents(prevContent, newContent))
			continue
		}
		merged = append(merged, msg)
	}
	return merged
}

func boundaryMarked(msg jval) bool {
	f, ok := msg.getField(preserveRoleBoundaryKey)
	return ok && f.kind == kindBool && f.s == "true"
}

func mergeContents(prev, next jval) jval {
	prevEmpty := contentEmpty(prev)
	nextEmpty := contentEmpty(next)
	if !prevEmpty && !nextEmpty {
		if prev.kind == kindArray || next.kind == kindArray {
			parts := append([]jval{}, asPartsList(prev)...)
			parts = append(parts, asPartsList(next)...)
			return jval{kind: kindArray, arr: parts}
		}
		return jstr(prev.s + "\n\n" + next.s)
	}
	if !nextEmpty {
		return next
	}
	return prev
}

func contentEmpty(v jval) bool {
	switch v.kind {
	case kindString:
		return v.s == ""
	case kindArray:
		return len(v.arr) == 0
	}
	return true
}

func asPartsList(v jval) []jval {
	if v.kind == kindArray {
		return v.arr
	}
	return []jval{jobj("type", jstr("text"), "text", jstr(v.s))}
}
