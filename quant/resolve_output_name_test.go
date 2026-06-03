// SPDX-License-Identifier: MIT OR Apache-2.0

package quant

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestResolveOutputNameParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "resolve_output_name.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		ModelName   string  `json:"model_name"`
		OQLevel     float64 `json:"oq_level"`
		Dtype       string  `json:"dtype"`
		PreserveMtp bool    `json:"preserve_mtp"`
		Result      string  `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		got := ResolveOutputName(c.ModelName, c.OQLevel, c.Dtype, c.PreserveMtp)
		if got != c.Result {
			t.Errorf("case %d: ResolveOutputName(%q, %v, %q, %v)\n got  %q\n want %q",
				i, c.ModelName, c.OQLevel, c.Dtype, c.PreserveMtp, got, c.Result)
		}
	}
}

func BenchmarkResolveOutputName(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = ResolveOutputName("Qwen3.5-122B-A10B-oQ6-fp16", 4, "float16", true)
	}
}
