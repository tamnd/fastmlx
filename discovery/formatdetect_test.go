// SPDX-License-Identifier: MIT OR Apache-2.0

package discovery

import (
	"encoding/json"
	"os"
	"testing"
)

type formatDetectFixture struct {
	Harmony []struct {
		Name      string  `json:"name"`
		ModelType *string `json:"model_type"`
		Out       bool    `json:"out"`
	} `json:"harmony"`
	Gemma4 []struct {
		Name      string  `json:"name"`
		ModelType *string `json:"model_type"`
		Out       bool    `json:"out"`
	} `json:"gemma4"`
	Qwen3 []struct {
		Name string `json:"name"`
		Out  bool   `json:"out"`
	} `json:"qwen3"`
}

func loadFormatDetectFixture(t *testing.T) formatDetectFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/formatdetect.json")
	if err != nil {
		t.Fatal(err)
	}
	var f formatDetectFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// strOrEmpty mirrors the reference's "no config or absent field" case: a JSON
// null model_type maps to the empty string the detectors treat as no config.
func strOrEmpty(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func TestIsHarmonyModelParity(t *testing.T) {
	fx := loadFormatDetectFixture(t)
	for _, c := range fx.Harmony {
		if got := IsHarmonyModel(c.Name, strOrEmpty(c.ModelType)); got != c.Out {
			t.Errorf("IsHarmonyModel(%q, %q) = %v, want %v", c.Name, strOrEmpty(c.ModelType), got, c.Out)
		}
	}
}

func TestIsGemma4ModelParity(t *testing.T) {
	fx := loadFormatDetectFixture(t)
	for _, c := range fx.Gemma4 {
		if got := IsGemma4Model(c.Name, strOrEmpty(c.ModelType)); got != c.Out {
			t.Errorf("IsGemma4Model(%q, %q) = %v, want %v", c.Name, strOrEmpty(c.ModelType), got, c.Out)
		}
	}
}

func TestIsQwen3ModelParity(t *testing.T) {
	fx := loadFormatDetectFixture(t)
	for _, c := range fx.Qwen3 {
		if got := IsQwen3Model(c.Name); got != c.Out {
			t.Errorf("IsQwen3Model(%q) = %v, want %v", c.Name, got, c.Out)
		}
	}
}

func BenchmarkIsHarmonyModel(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = IsHarmonyModel("mlx-community/gpt-oss-20b", "")
		_ = IsGemma4Model("google/gemma-4-9b", "")
		_ = IsQwen3Model("Qwen/Qwen3-8B")
	}
}
