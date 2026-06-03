// SPDX-License-Identifier: MIT OR Apache-2.0

package compute

// KV-cache bookkeeping mirrors mlx_lm.models.cache.KVCache and RotatingKVCache.
// These types track only the scalar index arithmetic — the offset, the
// allocated sequence capacity, and (for the rotating cache) the write cursor —
// that decides where each step's keys/values are written, when the backing
// buffer must grow or rotate, and which slice is fetched for attention. That
// arithmetic is pure and host-testable; it is the part that drives the GPU, not
// the GPU itself. The actual mx.zeros allocation, the concatenate, the
// `keys[..., a:b, :] = ...` assignment, and the returned array views are the cgo
// seam. A step's UpdatePlan carries exactly the ranges that seam needs to apply.
//
// Capacity here is keys.shape[2] in the reference (0 when keys is None). The
// step size (256) and the growth/rotation rules are reproduced verbatim so a Go
// model forward writes into the same layout the Python cache would.

// kvStep is KVCache.step / RotatingKVCache.step: the buffer grows in multiples
// of this many tokens.
const kvStep = 256

// UpdatePlan describes one update_and_fetch step in terms of buffer ranges. The
// cgo backend writes the step's keys/values into [WriteBegin, WriteEnd) and
// returns the views over [0, FetchEnd) (KVCache) or the rotated buffer
// (RotatingKVCache, where FetchEnd == Capacity once full). Grew reports whether
// the backing buffer was reallocated this step; TruncatedTail reports the
// KVCache case where an unaligned prior offset forced the slack tail to be
// sliced off before concatenating fresh zeros.
type UpdatePlan struct {
	Prev          int
	Offset        int
	Capacity      int
	WriteBegin    int
	WriteEnd      int
	FetchEnd      int
	Grew          bool
	TruncatedTail bool
}

// KVCache is the growable contiguous cache. Offset is the number of valid
// tokens; capacity is the allocated length.
type KVCache struct {
	Offset   int
	capacity int
}

// Capacity reports the allocated sequence length (keys.shape[2]; 0 when empty).
func (c *KVCache) Capacity() int { return c.capacity }

// Empty reports whether nothing has been allocated yet (keys is None).
func (c *KVCache) Empty() bool { return c.capacity == 0 }

// Size returns the number of valid tokens, matching KVCache.size().
func (c *KVCache) Size() int { return c.Offset }

// IsTrimmable matches KVCache.is_trimmable (always true).
func (c *KVCache) IsTrimmable() bool { return true }

// Update reproduces KVCache.update_and_fetch's bookkeeping for appending
// numSteps tokens. It reallocates in kvStep-sized blocks when the write would
// overrun the buffer, slicing off an unaligned tail first, then returns the
// write and fetch ranges. numSteps <= 0 is a no-op plan.
func (c *KVCache) Update(numSteps int) UpdatePlan {
	prev := c.Offset
	plan := UpdatePlan{Prev: prev}
	if numSteps <= 0 {
		plan.Offset, plan.Capacity, plan.FetchEnd = c.Offset, c.capacity, c.Offset
		plan.WriteBegin, plan.WriteEnd = prev, prev
		return plan
	}
	if c.capacity == 0 || prev+numSteps > c.capacity {
		plan.Grew = true
		nSteps := (kvStep + numSteps - 1) / kvStep
		block := nSteps * kvStep
		if c.capacity != 0 {
			if prev%kvStep != 0 {
				c.capacity = prev // slice the slack tail before concatenating
				plan.TruncatedTail = true
			}
			c.capacity += block
		} else {
			c.capacity = block
		}
	}
	c.Offset += numSteps
	plan.Offset = c.Offset
	plan.Capacity = c.capacity
	plan.WriteBegin, plan.WriteEnd = prev, c.Offset
	plan.FetchEnd = c.Offset
	return plan
}

// Trim drops up to n tokens from the end, matching KVCache.trim: it returns the
// number actually trimmed (min(offset, n)) and leaves capacity untouched.
func (c *KVCache) Trim(n int) int {
	n = min(n, c.Offset)
	c.Offset -= n
	return n
}

// RotatingKVCache is the sliding-window cache. It keeps the first Keep tokens
// pinned and rotates the remainder within a MaxSize window. Offset is the total
// tokens seen; Idx is the write cursor into the buffer; capacity is the
// allocated length.
type RotatingKVCache struct {
	Keep     int
	MaxSize  int
	Offset   int
	Idx      int
	capacity int
}

// NewRotatingKVCache constructs a rotating cache, matching
// RotatingKVCache(max_size, keep).
func NewRotatingKVCache(maxSize, keep int) *RotatingKVCache {
	return &RotatingKVCache{Keep: keep, MaxSize: maxSize}
}

// Capacity reports the allocated sequence length (keys.shape[2]; 0 when empty).
func (c *RotatingKVCache) Capacity() int { return c.capacity }

// Empty reports whether nothing has been allocated yet.
func (c *RotatingKVCache) Empty() bool { return c.capacity == 0 }

// Size returns min(offset, max_size), matching RotatingKVCache.size().
func (c *RotatingKVCache) Size() int {
	if c.Offset < c.MaxSize {
		return c.Offset
	}
	return c.MaxSize
}

// IsTrimmable matches RotatingKVCache.is_trimmable: only before the window fills.
func (c *RotatingKVCache) IsTrimmable() bool { return c.Offset < c.MaxSize }

// Update dispatches like update_and_fetch: a single-token step rotates in place,
// a multi-token step concatenates. It returns the write and fetch ranges.
func (c *RotatingKVCache) Update(numSteps int) UpdatePlan {
	if numSteps == 1 {
		return c.updateInPlace(1)
	}
	return c.updateConcat(numSteps)
}

// updateInPlace reproduces _update_in_place: grow toward MaxSize while there is
// room, trim down to the window once over it, rotate the cursor back to Keep at
// the boundary, then write the step.
func (c *RotatingKVCache) updateInPlace(s int) UpdatePlan {
	prev := c.Offset
	plan := UpdatePlan{Prev: prev}
	if c.capacity == 0 || (prev >= c.capacity && c.capacity < c.MaxSize) {
		plan.Grew = true
		newSize := min(kvStep, c.MaxSize-prev)
		if c.capacity != 0 {
			c.capacity += newSize
		} else {
			c.capacity = newSize
		}
		c.Idx = prev
	}
	if trimSize := c.capacity - c.MaxSize; trimSize > 0 {
		c.capacity = c.MaxSize
		c.Idx = c.MaxSize
	}
	if c.Idx == c.MaxSize {
		c.Idx = c.Keep
	}
	plan.WriteBegin, plan.WriteEnd = c.Idx, c.Idx+s
	c.Offset += s
	c.Idx += s
	plan.Offset = c.Offset
	plan.Capacity = c.capacity
	plan.FetchEnd = c.fetchEnd()
	return plan
}

// updateConcat reproduces _update_concat: put the buffer back in temporal order,
// trim to MaxSize+S-1 so every token keeps at least MaxSize of context, and
// append the step.
func (c *RotatingKVCache) updateConcat(s int) UpdatePlan {
	prev := c.Offset
	plan := UpdatePlan{Prev: prev}
	if c.capacity == 0 {
		c.capacity = s
	} else {
		c.capacity = c.temporalOrderCapacity()
		c.Idx = c.capacity
		if trimSize := c.Idx - c.MaxSize + 1; trimSize > 0 {
			c.capacity = c.capacity - trimSize + s
		} else {
			c.capacity += s
		}
	}
	c.Offset += s
	c.Idx = c.capacity
	plan.WriteBegin, plan.WriteEnd = prev, c.Offset
	plan.Offset = c.Offset
	plan.Capacity = c.capacity
	plan.FetchEnd = c.fetchEnd()
	return plan
}

// temporalOrderCapacity returns the buffer length after _temporal_order: a
// rearrange leaves the length unchanged, except an unfilled tail is sliced to
// the cursor.
func (c *RotatingKVCache) temporalOrderCapacity() int {
	if c.Idx == c.capacity || c.Idx < c.Offset {
		return c.capacity
	}
	return c.Idx
}

// fetchEnd returns the fetched length: the valid prefix while the window is
// unfilled, otherwise the whole buffer.
func (c *RotatingKVCache) fetchEnd() int {
	if c.Offset < c.MaxSize {
		return c.Offset
	}
	return c.capacity
}

// Trim drops up to n tokens, matching RotatingKVCache.trim: offset and the write
// cursor both move back by the trimmed count.
func (c *RotatingKVCache) Trim(n int) int {
	n = min(n, c.Offset)
	c.Offset -= n
	c.Idx -= n
	return n
}
