// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestValidateContextWindowParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "validate_context_window.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		NumPromptTokens int    `json:"num_prompt_tokens"`
		MaxCtxIsNone    bool   `json:"max_ctx_is_none"`
		MaxCtx          int    `json:"max_ctx"`
		Error           string `json:"error"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		var maxCtx *int
		if !c.MaxCtxIsNone {
			m := c.MaxCtx
			maxCtx = &m
		}
		err := ValidateContextWindow(c.NumPromptTokens, maxCtx)
		got := ""
		if err != nil {
			got = err.Error()
		}
		if got != c.Error {
			t.Errorf("case %d: ValidateContextWindow(%d, %v) error %q want %q",
				i, c.NumPromptTokens, maxCtx, got, c.Error)
		}
	}
}

func BenchmarkValidateContextWindow(b *testing.B) {
	b.ReportAllocs()
	m := 131072
	for b.Loop() {
		_ = ValidateContextWindow(200000, &m)
	}
}
