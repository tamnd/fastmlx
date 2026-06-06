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
	// leftPad carries the left padding this cache holds. On a merged batched cache
	// it is per-row: leftPad[b] is how many padding tokens were prepended to row b
	// so a short prompt's real tokens align, at the right edge, with the longest
	// prompt's. On a single-sequence cache it is the one-element slice of that
	// sequence's own padding, which is how a batch-prefilled row carries its dead
	// front positions across steps: mergeCachesAlongBatch reassembles the per-row
	// slice from each sequence's element and splitCachesAlongBatch writes each row's
	// element back. It is nil for a sequence with no padding and for a uniform
	// (equal-length) cohort, the synchronized path that needs neither per-row RoPE
	// offsets nor an explicit attention mask. SetLeftPad records it, normalizing an
	// all-zero slice to nil so those fast paths stay engaged.
	leftPad []int
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

// SetLeftPad records a ragged cohort's per-row left padding on the batched cache.
// A nil slice, or one whose entries are all zero, marks a uniform cohort and is
// normalized to nil so the uniform fast paths (scalar RoPE, built-in causal mask)
// stay engaged. The generator calls this once, at admission, after merging the
// cohort's caches and before the batched prefill, so every layer of the batched
// set shares the same padding description.
func (c *KVTensorCache) SetLeftPad(leftPad []int) {
	c.leftPad = nil
	for _, p := range leftPad {
		if p != 0 {
			c.leftPad = leftPad
			return
		}
	}
}

// LeftPad returns the per-row left padding recorded on the cache, or nil for a
// single sequence or uniform cohort.
func (c *KVTensorCache) LeftPad() []int { return c.leftPad }

// RopeOffsets returns the per-row RoPE position offset for a block decoded at the
// cache's current Offset, or nil when the cohort is uniform (every row shares the
// scalar Offset, the single-kernel RoPE fast path). For a left-padded cohort row b
// sits leftPad[b] positions behind the padded fill cursor, so its rope offset is
// Offset-leftPad[b]: the padding tokens prepended to a short prompt occupy the low
// positions that the real tokens must skip, which is the reference's per-row cache
// offset of -leftPad advanced by the tokens seen so far.
func (c *KVTensorCache) RopeOffsets() []int {
	if c.leftPad == nil {
		return nil
	}
	off := make([]int, len(c.leftPad))
	for b, pad := range c.leftPad {
		off[b] = c.Offset - pad
	}
	return off
}

// AttnMask returns what the attention SDPA should use for a block of L queries at
// this cache's offset: a mask mode string for SDPA's built-in path and an optional
// explicit additive mask. A uniform cohort reproduces the hardcoded behavior,
// ("causal", nil) for a multi-token prefill and ("", nil) for a single-token decode
// step. A left-padded cohort returns ("", mask) with the per-row additive mask built
// host-side by batchLeftPadCausalMask, shaped [batch, 1, L, offset+L]. The offset
// here is the pre-update offset and L is the new block, so offset+L is exactly the
// post-update key length the SDPA scores span, matching the reference
// create_causal_mask whose key axis is arange(offset + N). The one builder covers
// both cases: a prefill (L > 1) gets the full causal structure, a decode step
// (L == 1) gets a single query past every cached key so the causal term is vacuous
// and only the front-padding skip remains. The explicit mask carries both the causal
// structure and the padding skip, so the forward feeds it through sdpaWith with an
// empty mode in place of the built-in causal path.
func (c *KVTensorCache) AttnMask(batch, L int, s *mlxgo.Stream) (mode string, mask *mlxgo.Array, err error) {
	if c.leftPad == nil {
		if L > 1 {
			return "causal", nil, nil
		}
		return "", nil, nil
	}
	mask, err = batchLeftPadCausalMask(c.leftPad, L, c.Offset, s)
	return "", mask, err
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
// single offset describes every row. Each sequence's own left padding (the
// one-element leftPad a batch-prefilled row carries) is reassembled into the
// merged per-row slice, so a cohort prefilled together keeps masking its front
// padding on every subsequent batched decode; an all-zero slice normalizes to nil,
// keeping a no-padding cohort on the uniform fast path.
func mergeCachesAlongBatch(seqs [][]*KVTensorCache, s *mlxgo.Stream) ([]*KVTensorCache, error) {
	layers := len(seqs[0])
	merged := make([]*KVTensorCache, layers)
	pad := make([]int, len(seqs))
	for i, seq := range seqs {
		if lp := seq[0].leftPad; len(lp) > 0 {
			pad[i] = lp[0]
		}
	}
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
		mc.SetLeftPad(pad)
		merged[l] = mc
	}
	return merged, nil
}

// splitCachesAlongBatch writes a batched cache set back into the per-sequence
// caches after a batched forward has grown it. merged[l] holds the layer's batched
// key, value, and recurrent-state tensors and its advanced offset; this carves each
// back along the batch axis into seqs[i][l], so every sequence resumes with its own
// grown cache. Each row's own left padding travels with it as a one-element leftPad
// (nil when that row has none), so a sequence that prefilled in a cohort keeps
// masking its dead front positions whether its next step is batched or runs alone.
// It is the inverse of mergeCachesAlongBatch and runs only after the forward
// returns, so on the default stub the forward has already errored and this is never
// reached.
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
			p := 0
			if mc.leftPad != nil {
				p = mc.leftPad[i]
			}
			seq[l].SetLeftPad([]int{p})
		}
	}
	return nil
}
