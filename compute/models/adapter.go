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

// BatchLForward is the batch-polymorphic forward every model exposes as its
// unexported forwardBL: it runs batch rows of L tokens each (row-major, batch*L
// flat) through the model with a batched per-layer KVTensorCache set and returns
// the logits, shaped [batch, L, vocab]. The adapter drives it two ways, both with
// the cohort merged along the batch axis: a decode step passes L 1 (one token per
// row), a ragged prefill passes L the left-padded common width. Qwen3Model.forwardBL
// and the rest match it exactly, so a model value supplies the batched forward to
// NewAdapter without a wrapper.
type BatchLForward func(tokens []int32, batch, L int, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error)

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
	batchForward BatchLForward
}

// NewAdapter builds an adapter for a model with numLayers transformer layers, an
// end-of-sequence token id, its per-sequence forward, and its batch-polymorphic
// forward (the model's forwardBL). numLayers sizes the KV cache the generator
// allocates per sequence; it must equal the count both forwards expect (each
// rejects a mismatched cache length). batchForward is what compute.BatchGenerator
// calls for a synchronized cohort, at L 1 for a decode step and at the padded
// width for a ragged prefill; passing it here is what makes the adapter both a
// compute.BatchDecoder and a compute.BatchPrefiller.
func NewAdapter(numLayers, eos int, forward SequenceForward, batchForward BatchLForward) *Adapter {
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
	logits, err := a.batchForward(tokens, batch, 1, merged, s)
	if err != nil {
		return nil, err
	}
	if err := splitCachesAlongBatch(merged, seqs, s); err != nil {
		return nil, err
	}
	return batchRows(logits, batch, s)
}

// BatchPrefill runs a ragged cohort's prompts through the model in one
// left-padded batched forward and returns each sequence's logits at its own last
// real position. prompts[i] is sequence i's full prompt and caches[i] the fresh
// []*KVTensorCache that NewCache produced for it, in matching order.
//
// The prompts have different lengths, so they are left-padded to a common width:
// each row gets leftPadCohort's count of padding tokens prepended, which slides
// every prompt's real tokens to the right edge. With the last real token of every
// row aligned at the shared fill cursor, one forward decodes the whole cohort
// behind a single offset and a single gather at the right edge (lastRows) recovers
// each row's true final logits. The per-row left padding is recorded on the merged
// cache before the forward so its attention masks and RoPE offsets skip the front
// padding; after the forward the grown cache is split back into each sequence,
// every row advanced by the padded width. On the default stub the batched forward
// returns ErrMLXUnavailable at its first kernel, before the split and gather, the
// same wiring confirmation BatchDecode gives.
func (a *Adapter) BatchPrefill(prompts [][]int32, caches []any, s *mlxgo.Stream) ([]any, error) {
	batch := len(prompts)
	if batch == 0 {
		return nil, fmt.Errorf("models: batch prefill has no prompts")
	}
	if batch != len(caches) {
		return nil, fmt.Errorf("models: batch prefill has %d prompts but %d caches", batch, len(caches))
	}
	padded, leftPad, width := leftPadCohort(prompts)
	if width == 0 {
		return nil, fmt.Errorf("models: batch prefill has only empty prompts")
	}
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
	for _, mc := range merged {
		mc.SetLeftPad(leftPad)
	}
	logits, err := a.batchForward(padded, batch, width, merged, s)
	if err != nil {
		return nil, err
	}
	if err := splitCachesAlongBatch(merged, seqs, s); err != nil {
		return nil, err
	}
	return lastRows(logits, batch, s)
}

// leftPadCohort left-pads a ragged cohort of prompts to a common width and
// returns the row-major batch*width token buffer, the per-row left padding, and
// that width. The width is the longest prompt; a shorter row's leftPad is how many
// padding slots (zero, the embedding ignores them under the attention mask) are
// prepended so its real tokens land flush against the right edge, where every
// row's last real token then shares one position. An all-equal cohort gets an
// all-zero leftPad, which the cache normalizes to nil so the uniform fast path
// stays engaged. This is pure host arithmetic, the half of BatchPrefill that needs
// no kernel and carries the padding contract.
func leftPadCohort(prompts [][]int32) (padded []int32, leftPad []int, width int) {
	for _, p := range prompts {
		if len(p) > width {
			width = len(p)
		}
	}
	leftPad = make([]int, len(prompts))
	if width == 0 {
		return nil, leftPad, 0
	}
	padded = make([]int32, len(prompts)*width)
	for i, p := range prompts {
		leftPad[i] = width - len(p)
		copy(padded[i*width+leftPad[i]:(i+1)*width], p)
	}
	return padded, leftPad, width
}

// lastRows gathers a batched [batch, L, vocab] logits tensor into one [1, vocab]
// row per sequence, taken at the final position L-1. It is the prefill counterpart
// of batchRows: because the cohort is left-padded, every row's last real token sits
// at the right edge L-1, so one shared position recovers each row's true final
// logits. The gathers are the cgo seam; on the stub the batched forward already
// returned ErrMLXUnavailable, so this is never reached.
func lastRows(logits *mlxgo.Array, batch int, s *mlxgo.Stream) ([]any, error) {
	shape := logits.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("models: batched prefill logits have %d dims, want 3", len(shape))
	}
	last := shape[1] - 1
	posIdx, err := mlxgo.NewInt32([]int32{int32(last)}, 1)
	if err != nil {
		return nil, err
	}
	col, err := mlxgo.Take(logits, posIdx, 1, s) // [batch, 1, vocab]
	if err != nil {
		return nil, err
	}
	rows := make([]any, batch)
	for i := range batch {
		rowIdx, err := mlxgo.NewInt32([]int32{int32(i)}, 1)
		if err != nil {
			return nil, err
		}
		r, err := mlxgo.Take(col, rowIdx, 0, s) // [1, 1, vocab]
		if err != nil {
			return nil, err
		}
		rr, err := mlxgo.Reshape(r, []int{1, shape[2]}, s) // [1, vocab]
		if err != nil {
			return nil, err
		}
		rows[i] = rr
	}
	return rows, nil
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
// decodes a synchronized batch and prefills a ragged cohort, each in one forward.
var (
	_ compute.Model          = (*Adapter)(nil)
	_ compute.BatchDecoder   = (*Adapter)(nil)
	_ compute.BatchPrefiller = (*Adapter)(nil)
)
