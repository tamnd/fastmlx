// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"regexp"
	"strings"
)

// thinkTagPattern matches a full <think>...</think> span plus any trailing
// whitespace. The (?s) flag lets "." cross newlines and ".*?" is non-greedy so
// each span stops at its own close tag, matching the reference re.DOTALL
// pattern. No lookbehind or backreferences, so RE2 takes it verbatim.
var thinkTagPattern = regexp.MustCompile(`(?s)<think>.*?</think>\s*`)

// PreprocessHarmonyMessages prepares a message list for Harmony (gpt-oss)
// models. Assistant messages with string content have every <think>...</think>
// span stripped and the result trimmed, since the Harmony chat template expects
// the visible answer only. Tool, user, and system messages pass through
// unchanged (the chat template converts tool messages itself), as does an
// assistant message whose content is a non-string (e.g. a content-part list).
//
// The strip runs only when the content is a non-empty string that actually
// contains "<think>", so a clean answer keeps its exact bytes (no trim) and an
// empty content is left alone. Other fields on a modified message keep their
// original order with content replaced in place.
func PreprocessHarmonyMessages(messages []jval) []jval {
	result := make([]jval, 0, len(messages))
	for _, msg := range messages {
		if msg.kind != kindObject {
			// Mirror the reference, which skips any non-dict entry.
			continue
		}
		if msg.getString("role") == "assistant" {
			content := msg.getOr("content", jstr(""))
			if content.kind == kindString && content.s != "" && strings.Contains(content.s, "<think>") {
				cleaned := strings.TrimSpace(thinkTagPattern.ReplaceAllString(content.s, ""))
				msg = msg.setField("content", jstr(cleaned))
			}
		}
		result = append(result, msg)
	}
	return result
}
