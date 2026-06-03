// SPDX-License-Identifier: MIT OR Apache-2.0

package compute

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// namedBlob builds a minimal safetensors blob whose tensors are F32 scalars laid
// out contiguously, one per name. Enough to round-trip through LoadCheckpoint on
// the host, where only the names and shapes are read.
func namedBlob(names ...string) []byte {
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

func write(t *testing.T, dir, name string, data []byte) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), data, 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}

func TestLoadCheckpointSingleFile(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, ConfigFileName, []byte(`{"model_type":"qwen3","hidden_size":8}`))
	write(t, dir, "model.safetensors", namedBlob("a", "b", "c"))

	cfg, weights, err := LoadCheckpoint(dir)
	if err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}
	if !strings.Contains(string(cfg), `"model_type":"qwen3"`) {
		t.Fatalf("config not returned verbatim: %s", cfg)
	}
	assertNames(t, weights, "a", "b", "c")
}

func TestLoadCheckpointSharded(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, ConfigFileName, []byte(`{"model_type":"llama"}`))
	write(t, dir, "sa.safetensors", namedBlob("a", "b"))
	write(t, dir, "sb.safetensors", namedBlob("c"))
	// A weight in a file not named by the index must not be pulled in, so add a
	// stray file the index ignores.
	write(t, dir, "stray.safetensors", namedBlob("ignored"))
	index := `{"weight_map":{"a":"sa.safetensors","b":"sa.safetensors","c":"sb.safetensors"}}`
	write(t, dir, SafetensorsIndexName, []byte(index))

	_, weights, err := LoadCheckpoint(dir)
	if err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}
	assertNames(t, weights, "a", "b", "c")
	if _, ok := weights["ignored"]; ok {
		t.Fatal("loaded a weight from a file the index does not name")
	}
}

func TestLoadCheckpointMissingConfig(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "model.safetensors", namedBlob("a"))
	if _, _, err := LoadCheckpoint(dir); err == nil {
		t.Fatal("LoadCheckpoint accepted a directory with no config.json")
	}
}

func TestLoadCheckpointNoSafetensors(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, ConfigFileName, []byte(`{"model_type":"qwen3"}`))
	_, _, err := LoadCheckpoint(dir)
	if err == nil || !strings.Contains(err.Error(), "no safetensors") {
		t.Fatalf("err = %v, want a no-safetensors error", err)
	}
}

func TestLoadCheckpointDuplicateAcrossShards(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, ConfigFileName, []byte(`{"model_type":"qwen3"}`))
	write(t, dir, "sa.safetensors", namedBlob("w", "x"))
	write(t, dir, "sb.safetensors", namedBlob("w", "z")) // w duplicated
	index := `{"weight_map":{"x":"sa.safetensors","z":"sb.safetensors"}}`
	write(t, dir, SafetensorsIndexName, []byte(index))

	if _, _, err := LoadCheckpoint(dir); err == nil {
		t.Fatal("LoadCheckpoint accepted a weight duplicated across shards")
	}
}

func TestLoadCheckpointBadIndex(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, ConfigFileName, []byte(`{"model_type":"qwen3"}`))
	write(t, dir, SafetensorsIndexName, []byte(`{"weight_map":{}}`)) // empty map is invalid
	if _, _, err := LoadCheckpoint(dir); err == nil {
		t.Fatal("LoadCheckpoint accepted an empty index weight_map")
	}
}

func assertNames(t *testing.T, weights map[string]*mlxgo.Array, want ...string) {
	t.Helper()
	got := make([]string, 0, len(weights))
	for k := range weights {
		got = append(got, k)
	}
	sort.Strings(got)
	sort.Strings(want)
	if len(got) != len(want) {
		t.Fatalf("weights = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("weights = %v, want %v", got, want)
		}
	}
}

func BenchmarkLoadCheckpoint(b *testing.B) {
	dir := b.TempDir()
	os.WriteFile(filepath.Join(dir, ConfigFileName), []byte(`{"model_type":"qwen3"}`), 0o644)
	os.WriteFile(filepath.Join(dir, "model.safetensors"), namedBlob("a", "b", "c", "d", "e"), 0o644)
	b.ReportAllocs()
	for b.Loop() {
		if _, _, err := LoadCheckpoint(dir); err != nil {
			b.Fatalf("LoadCheckpoint: %v", err)
		}
	}
}
