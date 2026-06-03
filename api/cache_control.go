// SPDX-License-Identifier: MIT OR Apache-2.0

package api

// This file ports the cache-control probe that decides how an Anthropic
// response reports its input-side token usage. Anthropic's three input fields
// (input_tokens, cache_creation_input_tokens, cache_read_input_tokens) only
// partition the prompt when the client explicitly marks a region with
// cache_control; without that signal the cache fields stay at 0 and
// input_tokens carries the whole prompt count, regardless of any prefix caching
// the engine runs internally. It is a pure predicate over the request value.

// RequestHasCacheControl reports whether any system block, tool, or message
// content block in the request carries a truthy cache_control field. The system
// field is only scanned when it is a list of blocks (a bare system string never
// carries cache_control), tools are scanned when present, and message content
// is scanned only when it is a block list.
func RequestHasCacheControl(request jval) bool {
	if sys := request.getOr("system", jnull()); sys.kind == kindArray {
		for _, blk := range sys.arr {
			if blockHasCacheControl(blk) {
				return true
			}
		}
	}
	if tools := request.getOr("tools", jnull()); tools.kind == kindArray {
		for _, tool := range tools.arr {
			if blockHasCacheControl(tool) {
				return true
			}
		}
	}
	if messages := request.getOr("messages", jnull()); messages.kind == kindArray {
		for _, msg := range messages.arr {
			content := msg.getOr("content", jnull())
			if content.kind != kindArray {
				continue
			}
			for _, blk := range content.arr {
				if blockHasCacheControl(blk) {
					return true
				}
			}
		}
	}
	return false
}

// blockHasCacheControl reports whether a block carries a truthy cache_control
// field, mirroring the reference's getattr-and-test (an empty object or string
// is falsy, so it does not count).
func blockHasCacheControl(blk jval) bool {
	return pythonTruthy(blk.getOr("cache_control", jnull()))
}
