// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "strings"

// This file ports the structural half of the VLM image-extraction pass: walking
// OpenAI- and Responses-format chat messages, stripping image parts out of the
// content arrays into a flat ordered list of image references, and collapsing
// each message's text parts back into a plain string. The reference loads each
// reference into a PIL image inline; here the decode (HTTP fetch or data-URI /
// local-file open, EXIF transpose, RGB convert, and the drop-on-failure that
// goes with it) is the caller's seam, so this returns the resolved URLs in
// appearance order instead of decoded pixels. Extra message fields (tool_calls,
// tool_call_id, name, and the like) are preserved in their original order.

// ExtractImageRefsFromMessages returns the text-only messages alongside the
// image references found in their content arrays, in order of appearance. A
// message whose content is not an array passes through with its content
// coerced to a string (a falsy content becomes ""); a message whose content is
// an array has its text parts joined with newlines and its image_url /
// input_image parts resolved to URLs. Non-string, non-array content and parts
// without a usable url are dropped, matching the reference.
func ExtractImageRefsFromMessages(messages []jval) ([]jval, []string) {
	var textMessages []jval
	var urls []string
	for _, msg := range messages {
		role := msg.getOr("role", jstr("user"))
		content := msg.getOr("content", jstr(""))

		if content.kind != kindArray {
			if !pythonTruthy(content) {
				content = jstr("")
			}
			textMessages = append(textMessages, cleanedMessage(msg, role, content))
			continue
		}

		var textParts []string
		for _, part := range content.arr {
			if part.kind != kindObject {
				continue
			}
			switch part.getString("type") {
			case "text":
				if t := part.getOr("text", jnull()); pythonTruthy(t) {
					textParts = append(textParts, t.s)
				}
			case "image_url", "input_image":
				if url := resolveImageRefURL(part); url != "" {
					urls = append(urls, url)
				}
			}
		}
		textMessages = append(textMessages, cleanedMessage(msg, role, jstr(strings.Join(textParts, "\n"))))
	}
	return textMessages, urls
}

// resolveImageRefURL pulls the image URL out of an image_url / input_image part.
// The object is read from the image_url key, falling back to input_image when
// that is absent or null; a string holds the URL directly, an object holds it
// under url. Anything else yields "".
func resolveImageRefURL(part jval) string {
	obj := part.getOr("image_url", jnull())
	if obj.kind == kindNull {
		obj = part.getOr("input_image", jnull())
	}
	switch obj.kind {
	case kindString:
		return obj.s
	case kindObject:
		if u, ok := obj.getField("url"); ok && u.kind == kindString {
			return u.s
		}
	}
	return ""
}

// cleanedMessage rebuilds a message with role and content first, then every
// other field of the original in its original order.
func cleanedMessage(msg, role, content jval) jval {
	out := jobj("role", role, "content", content)
	for _, kv := range msg.obj {
		if kv.k == "role" || kv.k == "content" {
			continue
		}
		out = out.setField(kv.k, kv.v)
	}
	return out
}
