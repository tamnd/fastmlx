// SPDX-License-Identifier: MIT OR Apache-2.0

package quant

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type quantKeysFixture struct {
	Cases []struct {
		Input  map[string]any `json:"input"`
		Output map[string]any `json:"output"`
	} `json:"cases"`
}

func loadQuantKeys(t *testing.T) quantKeysFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/quantkeys.json")
	if err != nil {
		t.Fatal(err)
	}
	var f quantKeysFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestExpandPerLayerQuantKeysParity(t *testing.T) {
	for i, c := range loadQuantKeys(t).Cases {
		got := ExpandPerLayerQuantKeys(c.Input)
		if !reflect.DeepEqual(got, c.Output) {
			t.Errorf("case %d:\n got  %v\n want %v", i, got, c.Output)
		}
	}
}

func TestExpandPerLayerQuantKeysReturnsSameMap(t *testing.T) {
	cfg := map[string]any{"quantization": map[string]any{"lm_head": map[string]any{"bits": float64(8)}}}
	got := ExpandPerLayerQuantKeys(cfg)
	if !reflect.DeepEqual(got, cfg) {
		t.Errorf("expected the input map to be mutated in place and returned")
	}
	quant := cfg["quantization"].(map[string]any)
	if _, ok := quant["language_model.lm_head"]; !ok {
		t.Errorf("prefixed alias not added: %v", quant)
	}
}

func BenchmarkExpandPerLayerQuantKeys(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		cfg := map[string]any{
			"quantization": map[string]any{
				"lm_head":               map[string]any{"bits": float64(8)},
				"language_model.q_proj": map[string]any{"bits": float64(4)},
				"bits":                  float64(4),
			},
		}
		_ = ExpandPerLayerQuantKeys(cfg)
	}
}
