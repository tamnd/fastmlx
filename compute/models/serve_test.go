// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/fastmlx/compute"
	"github.com/tamnd/fastmlx/mlxgo"
	"github.com/tamnd/fastmlx/pipeline"
)

// fabricateBlob builds a minimal but valid safetensors container: every name is
// an F32 scalar, laid out contiguously. Construction stores the arrays without
// reading their values, so scalar placeholders are enough to assemble a model on
// the host; only the forward needs real weights and the GPU.
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

// nopSampler satisfies pipeline.Sampler. The forward reaches a kernel op before
// any token is sampled on the host, so it is never actually called.
type nopSampler struct{}

func (nopSampler) Sample(any) int { return 0 }

func TestNewBatchDecodeEndToEnd(t *testing.T) {
	args, err := ParseQwen3Args([]byte(minimalQwen3Config))
	if err != nil {
		t.Fatalf("ParseQwen3Args: %v", err)
	}
	blob := fabricateBlob(args.WeightNames())

	dec, err := NewBatchDecode([]byte(minimalQwen3Config), blob, 2)
	if err != nil {
		t.Fatalf("NewBatchDecode: %v", err)
	}
	// It is a drop-in decode strategy for the scheduler.
	var _ pipeline.DecodeStrategy = dec

	uid, err := dec.Insert(pipeline.DecodeRequest{Tokens: []int{1, 2, 3}, MaxTokens: 4, Sampler: nopSampler{}})
	if err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if !dec.HasActive() {
		t.Fatal("HasActive false after Insert")
	}
	// The scheduler's first Step runs the forward, which reaches a kernel op and
	// reports the missing backend: end-to-end wiring confirmed on the host.
	if _, err := dec.Step(); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("Step err = %v, want ErrMLXUnavailable", err)
	}
	if dec.Remove(uid) == nil {
		t.Fatal("Remove returned no cache for a live sequence")
	}
}

func TestNewBatchDecodeDirEndToEnd(t *testing.T) {
	args, err := ParseQwen3Args([]byte(minimalQwen3Config))
	if err != nil {
		t.Fatalf("ParseQwen3Args: %v", err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, compute.ConfigFileName), []byte(minimalQwen3Config), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), fabricateBlob(args.WeightNames()), 0o644); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	dec, err := NewBatchDecodeDir(dir, 2)
	if err != nil {
		t.Fatalf("NewBatchDecodeDir: %v", err)
	}
	if _, err := dec.Insert(pipeline.DecodeRequest{Tokens: []int{1, 2}, MaxTokens: 4, Sampler: nopSampler{}}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if _, err := dec.Step(); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("Step err = %v, want ErrMLXUnavailable", err)
	}
}

func TestNewBatchDecodeDirMissing(t *testing.T) {
	if _, err := NewBatchDecodeDir(filepath.Join(t.TempDir(), "nope"), 0); err == nil {
		t.Fatal("NewBatchDecodeDir accepted a missing directory")
	}
}

func TestNewBatchDecodeBadConfig(t *testing.T) {
	if _, err := NewBatchDecode([]byte(`{not json`), nil, 0); err == nil {
		t.Fatal("NewBatchDecode accepted invalid config JSON")
	}
}

func TestNewBatchDecodeNoModelType(t *testing.T) {
	// Valid empty blob so weight loading succeeds; the missing model_type is what
	// must fail.
	_, err := NewBatchDecode([]byte(`{"hidden_size":8}`), fabricateBlob(nil), 0)
	if err == nil || !strings.Contains(err.Error(), "model_type") {
		t.Fatalf("err = %v, want a no-model_type error", err)
	}
}

func TestNewBatchDecodeUnknownType(t *testing.T) {
	// Valid empty blob so weight loading succeeds; the dispatch is what fails.
	blob := fabricateBlob(nil)
	_, err := NewBatchDecode([]byte(`{"model_type":"nope"}`), blob, 0)
	if err == nil || !strings.Contains(err.Error(), "nope") {
		t.Fatalf("err = %v, want an unsupported-model_type error naming the type", err)
	}
}

func TestNewBatchDecodeBadBlob(t *testing.T) {
	// Known model_type but a truncated blob: the weight load must fail.
	if _, err := NewBatchDecode([]byte(`{"model_type":"qwen3"}`), []byte{0, 1, 2}, 0); err == nil {
		t.Fatal("NewBatchDecode accepted a truncated safetensors blob")
	}
}

func BenchmarkNewBatchDecode(b *testing.B) {
	args, _ := ParseQwen3Args([]byte(minimalQwen3Config))
	cfg := []byte(minimalQwen3Config)
	blob := fabricateBlob(args.WeightNames())
	b.ReportAllocs()
	for b.Loop() {
		if _, err := NewBatchDecode(cfg, blob, 2); err != nil {
			b.Fatalf("NewBatchDecode: %v", err)
		}
	}
}
