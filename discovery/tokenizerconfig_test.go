// SPDX-License-Identifier: MIT OR Apache-2.0

package discovery

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type tokenizerConfigFixture struct {
	GetTokenizerConfig []struct {
		ModelName       string         `json:"model_name"`
		TrustRemoteCode bool           `json:"trust_remote_code"`
		Out             map[string]any `json:"out"`
	} `json:"get_tokenizer_config"`
	ApplyQwen3Fix []struct {
		Config    map[string]any `json:"config"`
		ModelName string         `json:"model_name"`
		Out       map[string]any `json:"out"`
	} `json:"apply_qwen3_fix"`
}

func loadTokenizerConfig(t *testing.T) tokenizerConfigFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/tokenizerconfig.json")
	if err != nil {
		t.Fatal(err)
	}
	var f tokenizerConfigFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestGetTokenizerConfigParity(t *testing.T) {
	for i, c := range loadTokenizerConfig(t).GetTokenizerConfig {
		got := GetTokenizerConfig(c.ModelName, c.TrustRemoteCode)
		if !reflect.DeepEqual(got, c.Out) {
			t.Errorf("case %d (%q):\n got  %v\n want %v", i, c.ModelName, got, c.Out)
		}
	}
}

func TestApplyQwen3FixParity(t *testing.T) {
	for i, c := range loadTokenizerConfig(t).ApplyQwen3Fix {
		got := ApplyQwen3Fix(c.Config, c.ModelName)
		if !reflect.DeepEqual(got, c.Out) {
			t.Errorf("case %d (%q):\n got  %v\n want %v", i, c.ModelName, got, c.Out)
		}
	}
}

func BenchmarkGetTokenizerConfig(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = GetTokenizerConfig("Qwen3-8B", false)
	}
}
