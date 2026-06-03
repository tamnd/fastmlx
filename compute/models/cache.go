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
