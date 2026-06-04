// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"fmt"

	"github.com/tamnd/fastmlx/compute"
	"github.com/tamnd/fastmlx/mlxgo"
)

// SequenceForward is the per-sequence forward every model in this package
// exposes: it runs one sequence's tokens through the model with a per-layer
// KVTensorCache set and returns the logits, shaped [1, len(tokens), vocab].
// Qwen3Model.Forward, LlamaModel.Forward, and the rest match it exactly, so a
// model value is usable as one without a wrapper method.
type SequenceForward func(tokens []int32, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error)

// BatchSequenceForward is the batched-decode forward every model exposes
// alongside the per-sequence one: it runs batch decode tokens (one per row, the
// [batch] input) through the model with a batched per-layer KVTensorCache set and
// returns the logits, shaped [batch, 1, vocab]. Qwen3Model.BatchDecode and the
// rest match it exactly, so a model value supplies both forwards to NewAdapter
// without a wrapper.
type BatchSequenceForward func(tokens []int32, batch int, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error)

// Adapter wires a concrete model's per-sequence forward into compute.Model, the
// seam compute.BatchGenerator drives. It owns the two pieces the generator's
// any-typed seam keeps out of the bookkeeping core: allocating the per-sequence
// KV cache (one KVTensorCache per layer) and reducing the model's full
// [1, L, vocab] logits to the final-position row the sampler consumes.
//
// The adapter compiles and type-checks on the MLX-less host because it speaks
// only the mlxgo public API. NewCache and the cache-type guard run anywhere; the
// forward and the last-row slice are the cgo seam, returning ErrMLXUnavailable on
// the default build (the wrapped model's forward errors at its first kernel op,
// before the slice is reached) and dispatching Metal kernels under -tags mlx.
type Adapter struct {
	numLayers    int
	eos          int
	forward      SequenceForward
	batchForward BatchSequenceForward
}

// NewAdapter builds an adapter for a model with numLayers transformer layers, an
// end-of-sequence token id, its per-sequence forward, and its batched-decode
// forward. numLayers sizes the KV cache the generator allocates per sequence; it
// must equal the count both forwards expect (each rejects a mismatched cache
// length). batchForward is what compute.BatchGenerator calls for a synchronized
// cohort; passing it here is what makes the adapter a compute.BatchDecoder.
func NewAdapter(numLayers, eos int, forward SequenceForward, batchForward BatchSequenceForward) *Adapter {
	return &Adapter{numLayers: numLayers, eos: eos, forward: forward, batchForward: batchForward}
}

// NewCache allocates a fresh per-sequence KV cache: one growing KVTensorCache per
// layer, each empty until the sequence's first forward.
func (a *Adapter) NewCache() any {
	caches := make([]*KVTensorCache, a.numLayers)
	for i := range caches {
		caches[i] = &KVTensorCache{}
	}
	return caches
}

// Forward runs the model on this step's tokens and returns the final-position
// logits row, shaped [1, vocab], as the opaque value the sampler consumes. cache
// must be the []*KVTensorCache that NewCache produced for this sequence.
func (a *Adapter) Forward(tokens []int32, cache any, s *mlxgo.Stream) (any, error) {
	caches, ok := cache.([]*KVTensorCache)
	if !ok {
		return nil, fmt.Errorf("models: adapter cache is %T, want []*KVTensorCache", cache)
	}
	logits, err := a.forward(tokens, caches, s)
	if err != nil {
		return nil, err
	}
	return lastRow(logits, s)
}

// BatchDecode runs the model on a synchronized cohort's decode tokens in one
// forward and returns one final-position logits row per sequence, each shaped
// [1, vocab] as the opaque value a sampler consumes, in the caller's order. Each
// caches[i] must be the []*KVTensorCache that NewCache produced for sequence i; the
// adapter merges them along the batch axis, runs the batched forward (which grows
// the merged cache one step), splits the grown cache back into each sequence, and
// gathers the rows. On the default stub the batched forward returns
// ErrMLXUnavailable at its first kernel, before the split and gather, the same
// wiring confirmation the single-sequence Forward gives.
func (a *Adapter) BatchDecode(tokens []int32, caches []any, s *mlxgo.Stream) ([]any, error) {
	batch := len(caches)
	seqs := make([][]*KVTensorCache, batch)
	for i, c := range caches {
		kv, ok := c.([]*KVTensorCache)
		if !ok {
			return nil, fmt.Errorf("models: adapter cache %d is %T, want []*KVTensorCache", i, c)
		}
		seqs[i] = kv
	}
	merged, err := mergeCachesAlongBatch(seqs, s)
	if err != nil {
		return nil, err
	}
	logits, err := a.batchForward(tokens, batch, merged, s)
	if err != nil {
		return nil, err
	}
	if err := splitCachesAlongBatch(merged, seqs, s); err != nil {
		return nil, err
	}
	return batchRows(logits, batch, s)
}

// EOS reports the end-of-sequence token id that finishes a sequence.
func (a *Adapter) EOS() int { return a.eos }

// lastRow reduces [1, L, vocab] logits to the [1, vocab] row for the final
// position, matching the reference's logits[:, -1, :] before sampling. It gathers
// position L-1 along the sequence axis, then drops that axis. The single gather is
// the cgo seam; on the stub it returns ErrMLXUnavailable.
func lastRow(logits *mlxgo.Array, s *mlxgo.Stream) (*mlxgo.Array, error) {
	shape := logits.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("models: logits have %d dims, want 3", len(shape))
	}
	last := shape[1] - 1
	idx, err := mlxgo.NewInt32([]int32{int32(last)}, 1)
	if err != nil {
		return nil, err
	}
	row, err := mlxgo.Take(logits, idx, 1, s) // [1, 1, vocab]
	if err != nil {
		return nil, err
	}
	return mlxgo.Reshape(row, []int{shape[0], shape[2]}, s) // [1, vocab]
}

// batchRows gathers a batched [batch, 1, vocab] logits tensor into one [1, vocab]
// row per sequence, the batched counterpart of lastRow. The single decode timestep
// sits at position 0 on the sequence axis, so each row is just the batch slice for
// that sequence reshaped to drop the timestep. The gathers are the cgo seam; on the
// stub the batched forward already returned ErrMLXUnavailable, so this is never
// reached.
func batchRows(logits *mlxgo.Array, batch int, s *mlxgo.Stream) ([]any, error) {
	shape := logits.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("models: batched logits have %d dims, want 3", len(shape))
	}
	rows := make([]any, batch)
	for i := range batch {
		idx, err := mlxgo.NewInt32([]int32{int32(i)}, 1)
		if err != nil {
			return nil, err
		}
		row, err := mlxgo.Take(logits, idx, 0, s) // [1, 1, vocab]
		if err != nil {
			return nil, err
		}
		r, err := mlxgo.Reshape(row, []int{1, shape[2]}, s) // [1, vocab]
		if err != nil {
			return nil, err
		}
		rows[i] = r
	}
	return rows, nil
}

// compile-time check: the adapter is a BatchGenerator-ready model that also
// decodes a synchronized batch in one forward.
var (
	_ compute.Model        = (*Adapter)(nil)
	_ compute.BatchDecoder = (*Adapter)(nil)
)
