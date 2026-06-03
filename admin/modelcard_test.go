// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"testing"
)

type modelCardFixture struct {
	Version string `json:"version"`
	Today   string `json:"today"`
	Cases   []struct {
		ModelName        string         `json:"model_name"`
		Config           map[string]any `json:"config"`
		RedownloadNotice bool           `json:"redownload_notice"`
		Out              string         `json:"out"`
	} `json:"cases"`
}

func loadModelCard(t *testing.T) modelCardFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/modelcard.json")
	if err != nil {
		t.Fatal(err)
	}
	var f modelCardFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestGenerateModelCardParity(t *testing.T) {
	fx := loadModelCard(t)
	for i, c := range fx.Cases {
		got := GenerateModelCard(c.ModelName, c.Config, c.RedownloadNotice, fx.Today, fx.Version)
		if got != c.Out {
			t.Errorf("GenerateModelCard case %d (%s):\n got:\n%s\n want:\n%s", i, c.ModelName, got, c.Out)
		}
	}
}

func BenchmarkGenerateModelCard(b *testing.B) {
	config := map[string]any{
		"model_type":   "qwen3",
		"quantization": map[string]any{"bits": 4.0, "group_size": 64.0},
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = GenerateModelCard("Qwen3-30B-A3B-oQ4", config, true, "2026-06-03", "0.4.0")
	}
}
