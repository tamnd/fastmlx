// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import "github.com/tamnd/fastmlx/mlxgo"

// KVTensorCache is the array-backed key/value cache a model forward pass
// threads through its attention layers. Where compute.KVCache tracks only the
// scalar bookkeeping, this holds the actual cached key and value tensors and
// grows them as decoding proceeds.
//
// Update concatenates the new step's keys and values onto the running buffers
// along the sequence axis (axis 2 of the [batch, heads, seq, head_dim] layout)
// and returns the full cached tensors for attention to read. Offset carries the
// sequence length seen before the current step, which the attention layer feeds
// to RoPE as the position offset. This is the straightforward growing cache;
// the block-preallocating layout that compute.KVCache.Update plans is a later
// memory optimization that returns the identical key/value views.
//
// The type compiles in both builds because it speaks only the mlxgo public API.
// Under the default stub the first Update stores its inputs (no kernel) and a
// second Update returns ErrMLXUnavailable at the concatenate; under -tags mlx
// every Update runs on the GPU.
type KVTensorCache struct {
	Offset int
	keys   *mlxgo.Array
	values *mlxgo.Array
	// convState and ssmState hold a recurrent (gated delta net) layer's two
	// pieces of carried state: the depthwise convolution window and the per-head
	// recurrent state. Attention layers leave them nil and use keys/values; a
	// linear layer leaves keys/values nil and uses these. One cache type serves
	// both mixer kinds so the per-layer cache list stays a single slice.
	convState *mlxgo.Array
	ssmState  *mlxgo.Array
}

// Update appends keys and values to the cache and returns the full cached
// tensors. keys and values are shaped [batch, kvHeads, stepLen, headDim].
func (c *KVTensorCache) Update(keys, values *mlxgo.Array, s *mlxgo.Stream) (k, v *mlxgo.Array, err error) {
	if c.keys == nil {
		c.keys = keys
		c.values = values
	} else {
		c.keys, err = mlxgo.Concatenate([]*mlxgo.Array{c.keys, keys}, 2, s)
		if err != nil {
			return nil, nil, err
		}
		c.values, err = mlxgo.Concatenate([]*mlxgo.Array{c.values, values}, 2, s)
		if err != nil {
			return nil, nil, err
		}
	}
	c.Offset = c.keys.Shape()[2]
	return c.keys, c.values, nil
}

// Keys and Values expose the current cached tensors (nil before the first
// Update).
func (c *KVTensorCache) Keys() *mlxgo.Array   { return c.keys }
func (c *KVTensorCache) Values() *mlxgo.Array { return c.values }

// ConvState and SSMState expose a gated delta net layer's carried state (both
// nil before its first step).
func (c *KVTensorCache) ConvState() *mlxgo.Array { return c.convState }
func (c *KVTensorCache) SSMState() *mlxgo.Array  { return c.ssmState }

// SetState records a gated delta net step's new convolution window and recurrent
// state and advances the running sequence length by the step length. The linear
// mixer has no key/value tensors, so Offset is maintained here the way Update
// maintains it for an attention layer.
func (c *KVTensorCache) SetState(conv, ssm *mlxgo.Array, advance int) {
	c.convState = conv
	c.ssmState = ssm
	c.Offset += advance
}

// concatAlongBatch concatenates per-sequence arrays along the batch axis (axis 0)
// into the single tensor a batched forward consumes. A nil first element means the
// caches are still empty (no sequence has run a step yet, the state on the default
// stub where prefill stopped at the embedding), so it returns nil without a kernel;
// a single array passes through. This is the array-level primitive the cache merge
// builds on.
func concatAlongBatch(arrs []*mlxgo.Array, s *mlxgo.Stream) (*mlxgo.Array, error) {
	if len(arrs) == 0 || arrs[0] == nil {
		return nil, nil
	}
	if len(arrs) == 1 {
		return arrs[0], nil
	}
	return mlxgo.Concatenate(arrs, 0, s)
}

// splitAlongBatch divides a batched tensor back into n per-sequence arrays along
// the batch axis (axis 0), the inverse of concatAlongBatch. A nil input (an
// untouched empty cache) splits into n nils without a kernel.
func splitAlongBatch(a *mlxgo.Array, n int, s *mlxgo.Stream) ([]*mlxgo.Array, error) {
	if a == nil {
		return make([]*mlxgo.Array, n), nil
	}
	if n == 1 {
		return []*mlxgo.Array{a}, nil
	}
	return mlxgo.Split(a, n, 0, s)
}

// mergeCachesAlongBatch stacks per-sequence KV cache sets into one batched set the
// batched forward threads through its layers. seqs[i] is sequence i's per-layer
// cache list (all the same length); the result has one KVTensorCache per layer
// whose key, value, and recurrent-state tensors are the sequences concatenated
// along the batch axis, with the shared decode offset carried through. The
// generator only batches a synchronized cohort (every row at one offset), so a
// single offset describes every row.
func mergeCachesAlongBatch(seqs [][]*KVTensorCache, s *mlxgo.Stream) ([]*KVTensorCache, error) {
	layers := len(seqs[0])
	merged := make([]*KVTensorCache, layers)
	for l := range layers {
		keys := make([]*mlxgo.Array, len(seqs))
		values := make([]*mlxgo.Array, len(seqs))
		conv := make([]*mlxgo.Array, len(seqs))
		ssm := make([]*mlxgo.Array, len(seqs))
		for i, seq := range seqs {
			keys[i] = seq[l].keys
			values[i] = seq[l].values
			conv[i] = seq[l].convState
			ssm[i] = seq[l].ssmState
		}
		mc := &KVTensorCache{Offset: seqs[0][l].Offset}
		var err error
		if mc.keys, err = concatAlongBatch(keys, s); err != nil {
			return nil, err
		}
		if mc.values, err = concatAlongBatch(values, s); err != nil {
			return nil, err
		}
		if mc.convState, err = concatAlongBatch(conv, s); err != nil {
			return nil, err
		}
		if mc.ssmState, err = concatAlongBatch(ssm, s); err != nil {
			return nil, err
		}
		merged[l] = mc
	}
	return merged, nil
}

// splitCachesAlongBatch writes a batched cache set back into the per-sequence
// caches after a batched forward has grown it. merged[l] holds the layer's batched
// key, value, and recurrent-state tensors and its advanced offset; this carves each
// back along the batch axis into seqs[i][l], so every sequence resumes with its own
// grown cache. It is the inverse of mergeCachesAlongBatch and runs only after the
// forward returns, so on the default stub the forward has already errored and this
// is never reached.
func splitCachesAlongBatch(merged []*KVTensorCache, seqs [][]*KVTensorCache, s *mlxgo.Stream) error {
	n := len(seqs)
	for l, mc := range merged {
		keys, err := splitAlongBatch(mc.keys, n, s)
		if err != nil {
			return err
		}
		values, err := splitAlongBatch(mc.values, n, s)
		if err != nil {
			return err
		}
		conv, err := splitAlongBatch(mc.convState, n, s)
		if err != nil {
			return err
		}
		ssm, err := splitAlongBatch(mc.ssmState, n, s)
		if err != nil {
			return err
		}
		for i, seq := range seqs {
			seq[l].keys = keys[i]
			seq[l].values = values[i]
			seq[l].convState = conv[i]
			seq[l].ssmState = ssm[i]
			seq[l].Offset = mc.Offset
		}
	}
	return nil
}
