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
	numLayers int
	eos       int
	forward   SequenceForward
}

// NewAdapter builds an adapter for a model with numLayers transformer layers, an
// end-of-sequence token id, and its per-sequence forward. numLayers sizes the KV
// cache the generator allocates per sequence; it must equal the count the forward
// expects (the forward rejects a mismatched cache length).
func NewAdapter(numLayers, eos int, forward SequenceForward) *Adapter {
	return &Adapter{numLayers: numLayers, eos: eos, forward: forward}
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

// compile-time check: the adapter is a BatchGenerator-ready model.
var _ compute.Model = (*Adapter)(nil)
