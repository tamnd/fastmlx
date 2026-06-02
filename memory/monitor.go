// SPDX-License-Identifier: MIT OR Apache-2.0

// Package memory ports the GPU-free memory-estimation arithmetic the scheduler
// and cache use to size KV blocks and preflight prefill peaks. The live memory
// probing (MLX active memory, process RSS, the working-set ceiling) and the
// eviction enforcer are the compute seam and land with the backend; the
// estimators here are pure functions of the model dimensions and are exercised
// against reference-captured fixtures.
package memory

import "fmt"

// Monitor holds the model dimensions used for KV-cache memory estimation. In
// the paged SSD-only serving mode the KV data lives on disk, not GPU memory, so
// the pressure/free helpers are no-ops; the estimators feed block sizing and
// the prefill admission guard.
type Monitor struct {
	maxKVCacheMemory  int64
	numLayers         int
	numKVHeads        int
	headDim           int
	dtypeSize         int
	numAttentionHeads int
	numKVCacheLayers  int
}

// NewMonitor builds a monitor with the KV-cache memory ceiling (bytes). The
// ceiling must be positive, matching the reference constructor. The dtype size
// defaults to 2 (float16) until SetModelInfo overrides it.
func NewMonitor(maxKVCacheMemory int64) (*Monitor, error) {
	if maxKVCacheMemory <= 0 {
		return nil, fmt.Errorf("max_kv_cache_memory must be positive, got %d", maxKVCacheMemory)
	}
	return &Monitor{maxKVCacheMemory: maxKVCacheMemory, dtypeSize: 2}, nil
}

// SetModelInfo records the model dimensions for estimation. dtypeSize defaults
// to 2 (float16) when zero; numAttentionHeads defaults to numKVHeads; and
// numKVCacheLayers defaults to numLayers (it may be smaller for hybrid models
// where only some layers use full attention).
func (m *Monitor) SetModelInfo(numLayers, numKVHeads, headDim, dtypeSize, numAttentionHeads, numKVCacheLayers int) {
	m.numLayers = numLayers
	m.numKVHeads = numKVHeads
	m.headDim = headDim
	if dtypeSize > 0 {
		m.dtypeSize = dtypeSize
	} else {
		m.dtypeSize = 2
	}
	if numAttentionHeads > 0 {
		m.numAttentionHeads = numAttentionHeads
	} else {
		m.numAttentionHeads = numKVHeads
	}
	if numKVCacheLayers > 0 {
		m.numKVCacheLayers = numKVCacheLayers
	} else {
		m.numKVCacheLayers = numLayers
	}
}

// MaxKVCacheMemory returns the configured KV-cache memory ceiling.
func (m *Monitor) MaxKVCacheMemory() int64 { return m.maxKVCacheMemory }

// IsUnderPressure reports memory pressure. In paged SSD-only mode the KV cache
// lives on disk, so this is always false, matching the reference.
func (m *Monitor) IsUnderPressure() bool { return false }

// BytesToFree returns the bytes the cache should evict. In paged SSD-only mode
// there is nothing to free from GPU memory, so this is always 0.
func (m *Monitor) BytesToFree() int64 { return 0 }

// pick returns the first positive value, falling back to the last argument.
func pick(vals ...int) int {
	for _, v := range vals {
		if v > 0 {
			return v
		}
	}
	return 0
}

// EstimateBlockMemory estimates the bytes one KV-cache block of blockSize tokens
// uses: per layer, keys plus values shaped (1, kv_heads, block_size, head_dim).
// The override params use the stored value (or a ~7B default) when zero, matching
// the reference `arg or stored or default` chain.
func (m *Monitor) EstimateBlockMemory(blockSize, numLayers, numKVHeads, headDim, dtypeSize int) int64 {
	layers := pick(numLayers, m.numLayers, 32)
	kvHeads := pick(numKVHeads, m.numKVHeads, 8)
	dim := pick(headDim, m.headDim, 128)
	dtype := pick(dtypeSize, m.dtypeSize)
	perLayer := int64(blockSize) * int64(kvHeads) * int64(dim) * int64(dtype) * 2 // *2 for keys+values
	return perLayer * int64(layers)
}

// EstimatePromptKVBytes estimates the KV-cache bytes a prompt of numTokens uses.
// It uses the KVCache-layer count (hybrid models) falling back to the layer
// count, and returns 0 when the model dimensions are unset.
func (m *Monitor) EstimatePromptKVBytes(numTokens int) int64 {
	layers := pick(m.numKVCacheLayers, m.numLayers)
	kvHeads := m.numKVHeads
	dim := m.headDim
	dtype := m.dtypeSize
	if layers == 0 || kvHeads == 0 || dim == 0 {
		return 0
	}
	perToken := int64(layers) * int64(kvHeads) * int64(dim) * int64(dtype) * 2 // keys + values
	return int64(numTokens) * perToken
}

// EstimatePrefillPeakBytes estimates a request's prefill peak contribution: the
// newly allocated KV cache plus the SDPA attention activation peak for the last
// chunk, which attends over the full context (cachedTokens + newTokens). It
// returns 0 when the model dimensions are unset. The head_dim > 128 branch
// materializes the full float32 attention matrix (MLX fallback path); the
// head_dim <= 128 branch is the tiled fused kernel (output buffer only).
func (m *Monitor) EstimatePrefillPeakBytes(newTokens, chunkSize, cachedTokens int) int64 {
	hd := m.headDim
	nQ := m.numAttentionHeads
	if nQ == 0 || hd == 0 {
		return 0
	}
	attnSpan := int64(newTokens + cachedTokens)
	queryLen := int64(min(chunkSize, newTokens))
	var attn int64
	if hd > 128 {
		attn = int64(nQ) * queryLen * attnSpan * 4
		attn += int64(nQ) * queryLen * int64(hd) * 4 // output buffer (small)
	} else {
		attn = int64(nQ) * queryLen * int64(hd) * 4
	}
	return attn + m.EstimatePromptKVBytes(newTokens)
}

// EstimateChunkTransientBytes isolates the per-chunk SDPA attention transient
// for nTokens query tokens attending over kvLen context tokens, excluding the
// newly allocated KV (which becomes resident). It returns 0 when the model
// dimensions are unset or nTokens is non-positive.
func (m *Monitor) EstimateChunkTransientBytes(nTokens, kvLen int) int64 {
	hd := m.headDim
	nQ := m.numAttentionHeads
	if nQ == 0 || hd == 0 || nTokens <= 0 {
		return 0
	}
	if hd > 128 {
		return int64(nQ)*int64(nTokens)*int64(max(kvLen, 0))*4 + int64(nQ)*int64(nTokens)*int64(hd)*4
	}
	return int64(nQ) * int64(nTokens) * int64(hd) * 4
}

// EstimateBlocksToFree estimates how many blocks of blockSize tokens to evict to
// free bytesToFree bytes, rounded up, with a floor of 1 (0 when a block has no
// estimable size).
func (m *Monitor) EstimateBlocksToFree(bytesToFree int64, blockSize int) int64 {
	blockMem := m.EstimateBlockMemory(blockSize, 0, 0, 0, 0)
	if blockMem <= 0 {
		return 0
	}
	numBlocks := (bytesToFree + blockMem - 1) / blockMem // round up
	if numBlocks < 1 {
		return 1
	}
	return numBlocks
}
