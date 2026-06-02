// SPDX-License-Identifier: MIT OR Apache-2.0

package memory

import (
	"encoding/json"
	"os"
	"testing"
)

type memCfg struct {
	NumLayers         int `json:"num_layers"`
	NumKVHeads        int `json:"num_kv_heads"`
	HeadDim           int `json:"head_dim"`
	DtypeSize         int `json:"dtype_size"`
	NumAttentionHeads int `json:"num_attention_heads"`
	NumKVCacheLayers  int `json:"num_kv_cache_layers"`
}

func (c memCfg) monitor(t *testing.T) *Monitor {
	t.Helper()
	m, err := NewMonitor(1 << 40) // ceiling is irrelevant to the estimators
	if err != nil {
		t.Fatal(err)
	}
	m.SetModelInfo(c.NumLayers, c.NumKVHeads, c.HeadDim, c.DtypeSize, c.NumAttentionHeads, c.NumKVCacheLayers)
	return m
}

type memoryFixture struct {
	Block []struct {
		Cfg       memCfg `json:"cfg"`
		BlockSize int    `json:"block_size"`
		Expected  int64  `json:"expected"`
	} `json:"block"`
	PromptKV []struct {
		Cfg       memCfg `json:"cfg"`
		NumTokens int    `json:"num_tokens"`
		Expected  int64  `json:"expected"`
	} `json:"prompt_kv"`
	PrefillPeak []struct {
		Cfg          memCfg `json:"cfg"`
		NewTokens    int    `json:"new_tokens"`
		ChunkSize    int    `json:"chunk_size"`
		CachedTokens int    `json:"cached_tokens"`
		Expected     int64  `json:"expected"`
	} `json:"prefill_peak"`
	ChunkTransient []struct {
		Cfg      memCfg `json:"cfg"`
		NTokens  int    `json:"n_tokens"`
		KVLen    int    `json:"kv_len"`
		Expected int64  `json:"expected"`
	} `json:"chunk_transient"`
	BlocksToFree []struct {
		Cfg         memCfg `json:"cfg"`
		BytesToFree int64  `json:"bytes_to_free"`
		BlockSize   int    `json:"block_size"`
		Expected    int64  `json:"expected"`
	} `json:"blocks_to_free"`
}

func loadMemoryFixture(t *testing.T) memoryFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/memory.json")
	if err != nil {
		t.Fatal(err)
	}
	var f memoryFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestEstimationParity(t *testing.T) {
	fx := loadMemoryFixture(t)

	for i, c := range fx.Block {
		if got := c.Cfg.monitor(t).EstimateBlockMemory(c.BlockSize, 0, 0, 0, 0); got != c.Expected {
			t.Errorf("block[%d]: got %d want %d", i, got, c.Expected)
		}
	}
	for i, c := range fx.PromptKV {
		if got := c.Cfg.monitor(t).EstimatePromptKVBytes(c.NumTokens); got != c.Expected {
			t.Errorf("prompt_kv[%d]: got %d want %d", i, got, c.Expected)
		}
	}
	for i, c := range fx.PrefillPeak {
		got := c.Cfg.monitor(t).EstimatePrefillPeakBytes(c.NewTokens, c.ChunkSize, c.CachedTokens)
		if got != c.Expected {
			t.Errorf("prefill_peak[%d]: got %d want %d", i, got, c.Expected)
		}
	}
	for i, c := range fx.ChunkTransient {
		if got := c.Cfg.monitor(t).EstimateChunkTransientBytes(c.NTokens, c.KVLen); got != c.Expected {
			t.Errorf("chunk_transient[%d]: got %d want %d", i, got, c.Expected)
		}
	}
	for i, c := range fx.BlocksToFree {
		if got := c.Cfg.monitor(t).EstimateBlocksToFree(c.BytesToFree, c.BlockSize); got != c.Expected {
			t.Errorf("blocks_to_free[%d]: got %d want %d", i, got, c.Expected)
		}
	}
}

func TestNewMonitorRejectsNonPositiveCeiling(t *testing.T) {
	if _, err := NewMonitor(0); err == nil {
		t.Error("ceiling 0 should be rejected")
	}
	if _, err := NewMonitor(-1); err == nil {
		t.Error("negative ceiling should be rejected")
	}
}

func TestPagedSSDModeHelpers(t *testing.T) {
	m, _ := NewMonitor(1 << 30)
	if m.IsUnderPressure() {
		t.Error("paged SSD mode should never report pressure")
	}
	if m.BytesToFree() != 0 {
		t.Error("paged SSD mode should free nothing")
	}
	if m.MaxKVCacheMemory() != 1<<30 {
		t.Error("MaxKVCacheMemory should echo the ceiling")
	}
}

func TestEstimatorsZeroWhenModelInfoUnset(t *testing.T) {
	m, _ := NewMonitor(1 << 30) // no SetModelInfo
	if got := m.EstimatePromptKVBytes(1000); got != 0 {
		t.Errorf("prompt KV with no model info = %d, want 0", got)
	}
	if got := m.EstimatePrefillPeakBytes(1000, 2048, 0); got != 0 {
		t.Errorf("prefill peak with no model info = %d, want 0", got)
	}
	// Block memory still uses ~7B defaults when nothing is set.
	if got := m.EstimateBlockMemory(64, 0, 0, 0, 0); got == 0 {
		t.Error("block memory should use defaults when model info is unset")
	}
}

func BenchmarkEstimatePrefillPeakBytes(b *testing.B) {
	m, _ := NewMonitor(1 << 40)
	m.SetModelInfo(40, 8, 192, 2, 40, 20)
	b.ReportAllocs()
	for b.Loop() {
		_ = m.EstimatePrefillPeakBytes(4096, 2048, 8000)
	}
}
