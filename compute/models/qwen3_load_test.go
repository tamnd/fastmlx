// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

type qwen3CheckpointFixture struct {
	ConfigJSON            string `json:"config_json"`
	TiedBlobB64           string `json:"tied_blob_b64"`
	TiedWithLMHeadBlobB64 string `json:"tied_with_lmhead_blob_b64"`
	NumLayers             int    `json:"num_layers"`
	WeightCountTied       int    `json:"weight_count_tied"`
}

func loadCheckpointFixture(t *testing.T) qwen3CheckpointFixture {
	t.Helper()
	raw, err := os.ReadFile("../testdata/qwen3_checkpoint.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f qwen3CheckpointFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return f
}

func decodeB64(t *testing.T, s string) []byte {
	t.Helper()
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		t.Fatalf("decode base64: %v", err)
	}
	return b
}

func TestLoadQwen3EndToEnd(t *testing.T) {
	f := loadCheckpointFixture(t)
	m, err := LoadQwen3([]byte(f.ConfigJSON), decodeB64(t, f.TiedBlobB64))
	if err != nil {
		t.Fatalf("LoadQwen3: %v", err)
	}
	if len(m.layers) != f.NumLayers {
		t.Errorf("layers = %d, want %d", len(m.layers), f.NumLayers)
	}
	if m.embedTokens == nil || m.norm == nil {
		t.Error("embedTokens / norm not wired from the checkpoint")
	}
	if m.lmHead != nil {
		t.Error("tied checkpoint should leave lmHead nil")
	}
	// The loaded weights keep their on-disk bfloat16 dtype and real shapes.
	if got := m.embedTokens.Dtype(); got != mlxgo.BFloat16 {
		t.Errorf("embed dtype = %v, want bfloat16", got)
	}
	if shape := m.embedTokens.Shape(); len(shape) != 2 || shape[0] != 32 || shape[1] != 8 {
		t.Errorf("embed shape = %v, want [32 8]", shape)
	}
}

func TestLoadQwen3SanitizesStrayHead(t *testing.T) {
	// A tied checkpoint that still ships lm_head.weight must assemble: Sanitize
	// drops the stray head before NewQwen3Model would reject the extra key.
	f := loadCheckpointFixture(t)
	m, err := LoadQwen3([]byte(f.ConfigJSON), decodeB64(t, f.TiedWithLMHeadBlobB64))
	if err != nil {
		t.Fatalf("LoadQwen3 with stray head: %v", err)
	}
	if m.lmHead != nil {
		t.Error("stray lm_head should have been dropped for a tied model")
	}
}

func TestLoadQwen3BadConfig(t *testing.T) {
	f := loadCheckpointFixture(t)
	if _, err := LoadQwen3([]byte(`{`), decodeB64(t, f.TiedBlobB64)); err == nil {
		t.Error("expected a config decode error")
	}
}

func TestLoadQwen3BadBlob(t *testing.T) {
	f := loadCheckpointFixture(t)
	if _, err := LoadQwen3([]byte(f.ConfigJSON), []byte("not a safetensors blob")); err == nil {
		t.Error("expected a safetensors parse error")
	}
}
