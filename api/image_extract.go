// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "strings"

// This file ports the message-cleaning half of the VLM image-extraction step.
// The reference walks OpenAI-format messages, pulls every image part out of the
// content arrays, loads each image, and returns text-only messages alongside the
// loaded images. The image decode (a network/PIL fetch) is the seam: here the
// pure core returns the image URLs in order of appearance so the caller can load
// them, and the text-only messages it would have produced. They are pure
// transforms over the order-preserving jval model.

// ExtractImagesFromMessages strips image parts from OpenAI-format messages and
// returns the cleaned text-only messages plus the image URLs in appearance
// order. A message whose content is not an array passes through with its content
// coerced to "" when falsy. For an array content, text parts join with "\n" (an
// empty join yields ""), image_url/input_image parts contribute their URL, and
// every other part (including bare-string parts) is dropped. Extra message
// fields beyond role and content are preserved in their original order, and a
// missing role defaults to "user". The actual image load is the caller's seam;
// note the reference silently drops images that fail to load, so the loaded set
// may be shorter than the returned URL list.
func ExtractImagesFromMessages(messages []jval) (textMessages []jval, imageURLs []string) {
	textMessages = []jval{}
	imageURLs = []string{}
	for _, msg := range messages {
		role := msg.getOr("role", jstr("user"))
		content := msg.getOr("content", jstr(""))

		if content.kind != kindArray {
			coerced := content
			if !pythonTruthy(content) {
				coerced = jstr("")
			}
			textMessages = append(textMessages, rebuildMessage(role, coerced, msg))
			continue
		}

		var textParts []string
		for _, part := range content.arr {
			if part.kind != kindObject {
				continue
			}
			switch part.getString("type") {
			case "text":
				if tf, ok := part.getField("text"); ok && pythonTruthy(tf) && tf.kind == kindString {
					textParts = append(textParts, tf.s)
				}
			case "image_url", "input_image":
				if url := imagePartURL(part); url != "" {
					imageURLs = append(imageURLs, url)
				}
			}
		}
		joined := ""
		if len(textParts) > 0 {
			joined = strings.Join(textParts, "\n")
		}
		textMessages = append(textMessages, rebuildMessage(role, jstr(joined), msg))
	}
	return textMessages, imageURLs
}

// imagePartURL resolves the URL of an image part. It reads "image_url" first and
// falls back to "input_image" only when "image_url" is absent or null, mirroring
// the reference's `part.get("image_url")` then `is None` fallback. The resolved
// object is the URL when it is a bare string, its "url" member when it is an
// object, and "" otherwise.
func imagePartURL(part jval) string {
	obj, ok := part.getField("image_url")
	if !ok || obj.kind == kindNull {
		obj, _ = part.getField("input_image")
	}
	switch obj.kind {
	case kindString:
		return obj.s
	case kindObject:
		return obj.getString("url")
	}
	return ""
}

// rebuildMessage builds the cleaned message with role and content first, then
// every other field of the original message in its original order.
func rebuildMessage(role, content, original jval) jval {
	out := jval{kind: kindObject}
	out.obj = append(out.obj, jkv{"role", role}, jkv{"content", content})
	for _, kv := range original.obj {
		if kv.k != "role" && kv.k != "content" {
			out.obj = append(out.obj, jkv{kv.k, kv.v})
		}
	}
	return out
}
