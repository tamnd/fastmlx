// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestScaleAnthropicTokensParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "scale_anthropic_tokens.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		TokenCount        int  `json:"token_count"`
		ScalingEnabled    bool `json:"scaling_enabled"`
		TargetContextSize int  `json:"target_context_size"`
		ActualIsNone      bool `json:"actual_is_none"`
		Actual            int  `json:"actual"`
		Result            int  `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		var actual *int
		if !c.ActualIsNone {
			a := c.Actual
			actual = &a
		}
		got := ScaleAnthropicTokens(c.TokenCount, c.ScalingEnabled, c.TargetContextSize, actual)
		if got != c.Result {
			t.Errorf("case %d: ScaleAnthropicTokens(%d, %v, %d, %v) = %d want %d",
				i, c.TokenCount, c.ScalingEnabled, c.TargetContextSize, actual, got, c.Result)
		}
	}
}

func BenchmarkScaleAnthropicTokens(b *testing.B) {
	b.ReportAllocs()
	a := 32768
	for b.Loop() {
		_ = ScaleAnthropicTokens(50000, true, 200000, &a)
	}
}
