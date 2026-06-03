// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"regexp"
	"strings"
)

// toolRoleCheckRe matches a chat-template equality test against the "tool" role
// in either quote style, ported from _TOOL_ROLE_CHECK_RE in api/utils.py.
var toolRoleCheckRe = regexp.MustCompile(`==\s*['"]tool['"]`)

// ChatTemplateSupportsToolRole reports whether a tokenizer's chat template
// renders tool messages natively, ported from _chat_template_supports_tool_role
// in api/utils.py. The two tokenizer attributes it reads are injected so the
// predicate stays GPU- and toolkit-free: hasToolCalling is the tokenizer's own
// has_tool_calling flag, and chatTemplate is its chat_template string (nil when
// the tokenizer has no template or it is not a string, the reference's
// non-str-is-False case).
//
// It is a strict superset of has_tool_calling: a tokenizer that already flags
// itself wins immediately. Otherwise the template must both compare a role
// against "tool" (the regex) and reference the tool_calls variable; requiring
// both keeps a stray "tool" literal from counting as native support.
func ChatTemplateSupportsToolRole(hasToolCalling bool, chatTemplate *string) bool {
	if hasToolCalling {
		return true
	}
	if chatTemplate == nil {
		return false
	}
	if !toolRoleCheckRe.MatchString(*chatTemplate) {
		return false
	}
	return strings.Contains(*chatTemplate, "tool_calls")
}
