// SPDX-License-Identifier: MIT OR Apache-2.0

package main

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/fastmlx/compute"
	"github.com/tamnd/fastmlx/compute/models"
	"github.com/tamnd/fastmlx/pipeline"
)

// minimalQwen3Config is a valid two-layer qwen3 config: enough for the model to
// assemble on the host. The forward only runs once the scheduler calls Step, so
// no real weight values are needed to build the decode backend.
const minimalQwen3Config = `{
	"model_type": "qwen3",
	"hidden_size": 8,
	"num_hidden_layers": 2,
	"num_attention_heads": 2,
	"num_key_value_heads": 1,
	"head_dim": 4,
	"intermediate_size": 16,
	"vocab_size": 32,
	"rms_norm_eps": 1e-6,
	"rope_theta": 10000,
	"eos_token_id": 7
}`

// fabricateBlob builds a minimal valid safetensors container: every name is an
// F32 scalar laid out contiguously.
func fabricateBlob(names []string) []byte {
	type entry struct {
		Dtype       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets [2]int `json:"data_offsets"`
	}
	hdr := make(map[string]entry, len(names))
	for i, n := range names {
		hdr[n] = entry{Dtype: "F32", Shape: []int{1}, DataOffsets: [2]int{i * 4, (i + 1) * 4}}
	}
	hjson, _ := json.Marshal(hdr)
	out := make([]byte, 8+len(hjson)+len(names)*4)
	binary.LittleEndian.PutUint64(out[:8], uint64(len(hjson)))
	copy(out[8:], hjson)
	return out
}

func writeCheckpoint(t *testing.T, dir string) {
	t.Helper()
	args, err := models.ParseQwen3Args([]byte(minimalQwen3Config))
	if err != nil {
		t.Fatalf("ParseQwen3Args: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, compute.ConfigFileName), []byte(minimalQwen3Config), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), fabricateBlob(args.WeightNames()), 0o644); err != nil {
		t.Fatalf("write weights: %v", err)
	}
}

func TestDecodeBackendNoConfig(t *testing.T) {
	// An empty model directory has no checkpoint: the caller falls back to the
	// mock backend, signalled by a nil decode strategy.
	dec, name, err := decodeBackend(t.TempDir())
	if err != nil {
		t.Fatalf("decodeBackend: %v", err)
	}
	if dec != nil {
		t.Fatal("decode strategy is non-nil for a directory without a checkpoint")
	}
	if name != "mock-model" {
		t.Fatalf("model name = %q, want mock-model", name)
	}
}

func TestDecodeBackendValid(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "Qwen3-Tiny")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	writeCheckpoint(t, dir)

	dec, name, err := decodeBackend(dir)
	if err != nil {
		t.Fatalf("decodeBackend: %v", err)
	}
	if dec == nil {
		t.Fatal("decode strategy is nil for a valid checkpoint")
	}
	// It is a drop-in scheduler backend.
	var _ pipeline.DecodeStrategy = dec
	if name != "Qwen3-Tiny" {
		t.Fatalf("model name = %q, want the directory base Qwen3-Tiny", name)
	}
}

func TestDecodeBackendBrokenCheckpoint(t *testing.T) {
	// config.json present but no safetensors: a present-but-broken checkpoint is a
	// hard error, never a silent fall back to the mock backend.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, compute.ConfigFileName), []byte(minimalQwen3Config), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if _, _, err := decodeBackend(dir); err == nil {
		t.Fatal("decodeBackend accepted a checkpoint with no safetensors")
	}
}

func BenchmarkDecodeBackend(b *testing.B) {
	dir := b.TempDir()
	args, _ := models.ParseQwen3Args([]byte(minimalQwen3Config))
	_ = os.WriteFile(filepath.Join(dir, compute.ConfigFileName), []byte(minimalQwen3Config), 0o644)
	_ = os.WriteFile(filepath.Join(dir, "model.safetensors"), fabricateBlob(args.WeightNames()), 0o644)
	b.ReportAllocs()
	for b.Loop() {
		if _, _, err := decodeBackend(dir); err != nil {
			b.Fatalf("decodeBackend: %v", err)
		}
	}
}
