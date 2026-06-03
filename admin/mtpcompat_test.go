// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"testing"
)

type cacheSizeCase struct {
	Bytes int    `json:"bytes"`
	Out   string `json:"out"`
}

type paroCase struct {
	Config map[string]any `json:"config"`
	Out    []any          `json:"out"`
}

type compatCase struct {
	Config    map[string]any `json:"config"`
	ModelType *string        `json:"model_type"`
	Out       bool           `json:"out"`
}

type weightsCase struct {
	Keys []string `json:"keys"`
	Out  bool     `json:"out"`
}

type decisionCase struct {
	Config        map[string]any `json:"config"`
	HasMTPWeights bool           `json:"has_mtp_weights"`
	Compatible    bool           `json:"compatible"`
	Reason        string         `json:"reason"`
}

type mtpCompatFixture struct {
	Cache     []cacheSizeCase `json:"cache"`
	Paroquant []paroCase      `json:"paroquant"`
	Compat    []compatCase    `json:"compat"`
	Weights   []weightsCase   `json:"weights"`
	Decision  []decisionCase  `json:"decision"`
}

func loadMTPCompat(t *testing.T) mtpCompatFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/mtpcompat.json")
	if err != nil {
		t.Fatal(err)
	}
	var f mtpCompatFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestFormatCacheSizeParity(t *testing.T) {
	for i, c := range loadMTPCompat(t).Cache {
		if got := FormatCacheSize(c.Bytes); got != c.Out {
			t.Errorf("FormatCacheSize case %d (%d) = %q, want %q", i, c.Bytes, got, c.Out)
		}
	}
}

func TestIsParoquantConfigParity(t *testing.T) {
	for i, c := range loadMTPCompat(t).Paroquant {
		gotOK, gotReason := IsParoquantConfig(c.Config)
		wantOK := c.Out[0].(bool)
		wantReason := c.Out[1].(string)
		if gotOK != wantOK || gotReason != wantReason {
			t.Errorf("IsParoquantConfig case %d (%v) = (%v, %q), want (%v, %q)", i, c.Config, gotOK, gotReason, wantOK, wantReason)
		}
	}
}

func TestIsMTPCompatibleParity(t *testing.T) {
	for i, c := range loadMTPCompat(t).Compat {
		mt := ""
		if c.ModelType != nil {
			mt = *c.ModelType
		}
		if got := IsMTPCompatible(c.Config, mt); got != c.Out {
			t.Errorf("IsMTPCompatible case %d (%v, %v) = %v, want %v", i, c.Config, c.ModelType, got, c.Out)
		}
	}
}

func TestModelHasMTPWeightTensorsParity(t *testing.T) {
	for i, c := range loadMTPCompat(t).Weights {
		if got := ModelHasMTPWeightTensors(c.Keys); got != c.Out {
			t.Errorf("ModelHasMTPWeightTensors case %d (%v) = %v, want %v", i, c.Keys, got, c.Out)
		}
	}
}

func TestMTPCompatForModelParity(t *testing.T) {
	for i, c := range loadMTPCompat(t).Decision {
		gotOK, gotReason := MTPCompatForModel(c.Config, c.HasMTPWeights)
		if gotOK != c.Compatible || gotReason != c.Reason {
			t.Errorf("MTPCompatForModel case %d (%v):\n got  (%v, %q)\n want (%v, %q)", i, c.Config, gotOK, gotReason, c.Compatible, c.Reason)
		}
	}
}

func BenchmarkMTPCompatForModel(b *testing.B) {
	config := map[string]any{"mtp_num_hidden_layers": 1.0, "model_type": "qwen3_6"}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = MTPCompatForModel(config, true)
	}
}
