// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestChatTemplateSupportsToolRoleParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "chat_template_tool_role.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Label             string  `json:"label"`
		HasToolCalling    bool    `json:"has_tool_calling"`
		ChatTemplate      *string `json:"chat_template"`
		ChatTemplateIsStr bool    `json:"chat_template_is_string"`
		Result            bool    `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		// A non-string template is the reference's non-str-is-False case; model
		// it as a nil template so the predicate sees "no usable template".
		tmpl := c.ChatTemplate
		if !c.ChatTemplateIsStr {
			tmpl = nil
		}
		got := ChatTemplateSupportsToolRole(c.HasToolCalling, tmpl)
		if got != c.Result {
			t.Errorf("case %d (%s): got %v want %v", i, c.Label, got, c.Result)
		}
	}
}

func BenchmarkChatTemplateSupportsToolRole(b *testing.B) {
	b.ReportAllocs()
	tmpl := `{% if message.role == "tool" %}{{ message.tool_calls }}{% endif %}`
	for b.Loop() {
		_ = ChatTemplateSupportsToolRole(false, &tmpl)
	}
}
