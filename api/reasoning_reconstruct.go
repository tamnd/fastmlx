// SPDX-License-Identifier: MIT OR Apache-2.0

package api

// This file ports the reasoning-reconstruction step that runs while extracting
// text from a historical assistant message. External clients echo a model's
// chain of thought back through the OpenAI reasoning_content field (or Anthropic
// thinking blocks), and chat templates handle it two ways: a native template
// reads a top-level reasoning_content field and wants the content left clean,
// while a non-native template only understands <think>...</think> embedded in
// the content, so the reasoning is inlined as a fallback. It is a pure transform
// over the order-preserving jval model; whether the active template is native is
// decided upstream from the tokenizer and passed in as a flag.

// applyReasoningReconstruction reconstructs reasoning on a historical assistant
// message. It returns the new content, the reasoning string to attach as a
// reasoning_content field, and whether that reasoning string should be attached
// at all. Only an assistant message with non-empty reasoning is touched; any
// other message passes through with no reasoning to attach. Content that is a
// content-block list is reduced to its text parts first. When native, the
// content stays clean and the reasoning travels separately; otherwise the
// reasoning is inlined ahead of the content inside a <think> block.
func applyReasoningReconstruction(role string, content jval, reasoning string, native bool) (jval, string, bool) {
	if role != "assistant" || reasoning == "" {
		return content, "", false
	}
	text := ""
	switch content.kind {
	case kindString:
		text = content.s
	case kindArray:
		text = ExtractTextFromContentList(content)
	}
	if native {
		return jstr(text), reasoning, true
	}
	return jstr("<think>\n" + reasoning + "\n</think>\n\n" + text), "", false
}
