// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"regexp"
	"strings"
)

// This file ports the output post-processing helpers: stripping special tokens
// from model text, splitting off thinking blocks, the partial-mode detection
// that runs before the chat template, and the token-usage alias normalization
// that keeps the OpenAI and Anthropic field names in sync.

// specialTokensPattern matches the special tokens removed from model output. The
// alternation is reproduced verbatim from the reference (the inner |...| are
// literal pipes, escaped; the bare | separate alternatives).
var specialTokensPattern = regexp.MustCompile(
	`<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|` +
		`<\|end\|>|<\|eot_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|` +
		`</s>|<s>|<pad>|\[PAD\]|\[SEP\]|\[CLS\]`)

// CleanSpecialTokens removes only special tokens, preserving <think>...</think>
// blocks for downstream processing, and trims surrounding whitespace. Empty
// input is returned unchanged.
func CleanSpecialTokens(text string) string {
	if text == "" {
		return text
	}
	return strings.TrimSpace(specialTokensPattern.ReplaceAllString(text, ""))
}

// CleanOutputText removes special tokens and then strips thinking blocks,
// returning the trimmed visible content. Empty input is returned unchanged.
func CleanOutputText(text string) string {
	if text == "" {
		return text
	}
	stripped := specialTokensPattern.ReplaceAllString(text, "")
	_, content := ExtractThinking(stripped)
	return strings.TrimSpace(content)
}

// DetectAndStripPartial reports whether the final message is an assistant
// message flagged partial=true, and returns the messages with the partial key
// removed from every one (it is not part of the chat-template contract). The
// input messages are not mutated; cleaned copies are returned.
func DetectAndStripPartial(messages []jval) (bool, []jval) {
	isPartial := false
	if n := len(messages); n > 0 {
		last := messages[n-1]
		if last.getString("role") == "assistant" {
			if p, ok := last.getField("partial"); ok && pythonTruthy(p) {
				isPartial = true
			}
		}
	}
	cleaned := make([]jval, len(messages))
	for i, msg := range messages {
		cleaned[i] = msg.removeField("partial")
	}
	return isPartial, cleaned
}

// removeField returns a copy of an object value with the named key removed. A
// non-object value is returned unchanged.
func (v jval) removeField(key string) jval {
	if v.kind != kindObject {
		return v
	}
	out := jval{kind: kindObject}
	for _, kv := range v.obj {
		if kv.k == key {
			continue
		}
		out.obj = append(out.obj, kv)
	}
	return out
}

// BaseUsage tracks token counts in both OpenAI (prompt/completion) and Anthropic
// (input/output) naming.
type BaseUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	InputTokens      int
	OutputTokens     int
}

// NormalizeBaseUsage fills in total_tokens from prompt+completion when it is
// unset, and mirrors prompt/completion into the Anthropic-style input/output
// aliases when those are unset. This matches the reference's post-init sync:
// each derived field is only filled when it is still zero and its source is
// positive.
func NormalizeBaseUsage(u BaseUsage) BaseUsage {
	if u.TotalTokens == 0 && (u.PromptTokens > 0 || u.CompletionTokens > 0) {
		u.TotalTokens = u.PromptTokens + u.CompletionTokens
	}
	if u.InputTokens == 0 && u.PromptTokens > 0 {
		u.InputTokens = u.PromptTokens
	}
	if u.OutputTokens == 0 && u.CompletionTokens > 0 {
		u.OutputTokens = u.CompletionTokens
	}
	return u
}
