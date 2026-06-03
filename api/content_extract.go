// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "strings"

// This file ports the content-array extraction helpers that flatten OpenAI- and
// Anthropic-style content lists into the shapes the chat template and the VLM
// path consume, plus the void-assistant-message filter. They are pure transforms
// over the order-preserving jval model. The reference also accepts Pydantic
// models (model_dump); here the input is already decoded JSON, so only the dict
// and string item shapes apply.

// ExtractTextFromContentList joins the text parts of a content array, dropping
// non-text items. An object item with type "text" contributes its "text" field
// (empty when absent); a bare string contributes itself. Parts join with "\n";
// an empty result is the empty string.
func ExtractTextFromContentList(content jval) string {
	if content.kind != kindArray {
		return ""
	}
	var parts []string
	for _, item := range content.arr {
		switch item.kind {
		case kindObject:
			if item.getString("type") == "text" {
				parts = append(parts, item.getString("text"))
			}
		case kindString:
			parts = append(parts, item.s)
		}
	}
	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, "\n")
}

// ExtractMultimodalContentList keeps text and image parts of a content array,
// normalizing them to OpenAI shapes for VLM processing and dropping everything
// else. Text and input_text become {type:text, text:...} (the text falls back to
// the "content" field, then ""); image_url/input_image normalize to
// {type:image_url, image_url:{url}}; Anthropic base64 image becomes a data URL.
// Items whose resolved url is empty are dropped.
func ExtractMultimodalContentList(content jval) jval {
	parts := []jval{}
	if content.kind != kindArray {
		return jval{kind: kindArray, arr: parts}
	}
	for _, item := range content.arr {
		if item.kind != kindObject {
			continue
		}
		switch item.getString("type") {
		case "text", "input_text":
			text := firstTruthyField(item, "text", "content")
			parts = append(parts, jobj("type", jstr("text"), "text", text))
		case "image_url":
			if url, ok := resolveImageURL(item, "image_url"); ok {
				parts = append(parts, imageURLPart(jstr(url)))
			}
		case "input_image":
			key := "image_url"
			if !item.hasField("image_url") {
				key = "input_image"
			}
			if url, ok := resolveImageURL(item, key); ok {
				parts = append(parts, imageURLPart(jstr(url)))
			}
		case "image":
			source, _ := item.getField("source")
			if source.kind == kindObject && source.getString("type") == "base64" {
				mediaType := source.getString("media_type")
				if mediaType == "" {
					mediaType = "image/jpeg"
				}
				data := source.getString("data")
				url := "data:" + mediaType + ";base64," + data
				parts = append(parts, imageURLPart(jstr(url)))
			}
		}
	}
	return jval{kind: kindArray, arr: parts}
}

// firstTruthyField returns the first of the named fields whose value is truthy,
// or an empty string when none is. Mirrors Python's `a or b or ""` chain.
func firstTruthyField(item jval, keys ...string) jval {
	for _, k := range keys {
		if f, ok := item.getField(k); ok && pythonTruthy(f) {
			return f
		}
	}
	return jstr("")
}

// resolveImageURL reads a url from a field that may be a bare string or an
// object with a "url" member, reporting false when the url is empty/absent.
func resolveImageURL(item jval, key string) (string, bool) {
	v, ok := item.getField(key)
	if !ok {
		return "", false
	}
	var url string
	switch v.kind {
	case kindString:
		url = v.s
	case kindObject:
		url = v.getString("url")
	}
	if url == "" {
		return "", false
	}
	return url, true
}

func imageURLPart(url jval) jval {
	return jobj("type", jstr("image_url"), "image_url", jobj("url", url))
}

// DropVoidAssistantMessages removes assistant messages that carry no content,
// tool_calls, tool_responses, or reasoning_content. Such messages make strict
// chat templates raise; the carried payloads keep an otherwise-empty assistant
// turn alive.
func DropVoidAssistantMessages(messages []jval) []jval {
	out := make([]jval, 0, len(messages))
	for _, msg := range messages {
		void := msg.getString("role") == "assistant" &&
			!fieldTruthy(msg, "content") &&
			!fieldTruthy(msg, "tool_calls") &&
			!fieldTruthy(msg, "tool_responses") &&
			!fieldTruthy(msg, "reasoning_content")
		if !void {
			out = append(out, msg)
		}
	}
	return out
}

// fieldTruthy reports whether the named field is present and truthy.
func fieldTruthy(msg jval, key string) bool {
	f, ok := msg.getField(key)
	return ok && pythonTruthy(f)
}
