// SPDX-License-Identifier: MIT OR Apache-2.0

//go:build !mlx

package mlxgo

// This is the default, GPU-free build of the binding. An Array holds its shape,
// dtype, and (for arrays constructed from Go data) the host bytes, so the
// metadata and host round-trip parts of the API work without MLX. Every
// operation that would dispatch a kernel returns ErrMLXUnavailable. The cgo
// build (-tags mlx) replaces this file with the real mlx-c surface and exports
// the identical API.

// Array mirrors the cgo build's Array. In the stub it carries only host-side
// metadata and any data it was constructed with.
type Array struct {
	shape []int
	dtype Dtype
	data  any // []float32, []int32, or nil
	freed bool
}

// NewFloat32 builds a float32 array from host data and a shape. The data length
// must equal the product of the shape.
func NewFloat32(data []float32, shape ...int) (*Array, error) {
	if len(data) != elementCount(shape) {
		return nil, errShape("NewFloat32", len(data), shape)
	}
	cp := make([]float32, len(data))
	copy(cp, data)
	return &Array{shape: cloneShape(shape), dtype: Float32, data: cp}, nil
}

// NewInt32 builds an int32 array from host data and a shape.
func NewInt32(data []int32, shape ...int) (*Array, error) {
	if len(data) != elementCount(shape) {
		return nil, errShape("NewInt32", len(data), shape)
	}
	cp := make([]int32, len(data))
	copy(cp, data)
	return &Array{shape: cloneShape(shape), dtype: Int32, data: cp}, nil
}

// NewBytes builds an array of any dtype from its raw little-endian element
// bytes and a shape. This is the loader path: safetensors weights arrive as a
// byte range with a dtype tag (including float16 and bfloat16, which have no Go
// scalar), and the cgo build hands those bytes straight to mlx. The data length
// must equal the element count times the dtype's element size.
func NewBytes(data []byte, dtype Dtype, shape ...int) (*Array, error) {
	if !dtype.Valid() || dtype.Size() == 0 {
		return nil, errShape("NewBytes", len(data), shape)
	}
	if len(data) != elementCount(shape)*dtype.Size() {
		return nil, errShape("NewBytes", len(data), shape)
	}
	cp := make([]byte, len(data))
	copy(cp, data)
	return &Array{shape: cloneShape(shape), dtype: dtype, data: cp}, nil
}

// Zeros builds a zero-filled array of the given dtype and shape.
func Zeros(dtype Dtype, shape ...int) (*Array, error) {
	if !dtype.Valid() {
		return nil, ErrMLXUnavailable
	}
	a := &Array{shape: cloneShape(shape), dtype: dtype}
	switch dtype {
	case Float32:
		a.data = make([]float32, elementCount(shape))
	case Int32:
		a.data = make([]int32, elementCount(shape))
	}
	return a, nil
}

// Shape returns a copy of the array's shape.
func (a *Array) Shape() []int { return cloneShape(a.shape) }

// Dtype returns the array's element type.
func (a *Array) Dtype() Dtype { return a.dtype }

// Ndim returns the number of dimensions.
func (a *Array) Ndim() int { return len(a.shape) }

// Size returns the total number of elements.
func (a *Array) Size() int { return elementCount(a.shape) }

// Eval is a no-op in the stub (there is no deferred graph to materialize).
func (a *Array) Eval() error {
	if a.freed {
		return ErrMLXUnavailable
	}
	return nil
}

// ToFloat32 returns the host float32 data of an array constructed from float32
// data; any other case (no host data, or a kernel result that never ran) errors.
func (a *Array) ToFloat32() ([]float32, error) {
	if a.freed {
		return nil, ErrMLXUnavailable
	}
	if d, ok := a.data.([]float32); ok {
		out := make([]float32, len(d))
		copy(out, d)
		return out, nil
	}
	return nil, ErrMLXUnavailable
}

// ToInt32 returns the host int32 data of an int32 array.
func (a *Array) ToInt32() ([]int32, error) {
	if a.freed {
		return nil, ErrMLXUnavailable
	}
	if d, ok := a.data.([]int32); ok {
		out := make([]int32, len(d))
		copy(out, d)
		return out, nil
	}
	return nil, ErrMLXUnavailable
}

// Free releases the array. In the stub it just drops the host data.
func (a *Array) Free() {
	a.data = nil
	a.freed = true
}

// Stream mirrors the cgo Stream. In the stub it is an inert handle.
type Stream struct{ id int }

// DefaultStream returns the default device stream.
func DefaultStream() *Stream { return &Stream{} }

// NewStream creates a new stream on the default device.
func NewStream() (*Stream, error) { return &Stream{}, nil }

// NewThreadLocalStream creates a stream pinned to the calling thread, mirroring
// the per-engine Metal stream the serving layer dedicates to its step thread.
func NewThreadLocalStream() (*Stream, error) { return &Stream{}, nil }

// Synchronize blocks until the stream's queued work completes (a no-op here).
func (s *Stream) Synchronize() error { return nil }

// SetDefaultStream makes s the default stream (a no-op here).
func SetDefaultStream(s *Stream) {}

// ClearCache releases MLX's cached buffers (a no-op here).
func ClearCache() {}

// SetWiredLimit sets the Metal wired-memory limit in bytes (a no-op here).
func SetWiredLimit(bytes uint64) {}

// SetMemoryLimit sets the soft memory limit in bytes (a no-op here).
func SetMemoryLimit(bytes uint64) {}

// SetCacheLimit sets the buffer-cache limit in bytes (a no-op here).
func SetCacheLimit(bytes uint64) {}

// GetActiveMemory reports MLX's active memory in bytes (always 0 here).
func GetActiveMemory() uint64 { return 0 }

// GetPeakMemory reports MLX's peak memory in bytes (always 0 here).
func GetPeakMemory() uint64 { return 0 }

// The compute operations. Each returns ErrMLXUnavailable in the stub; the cgo
// build dispatches the corresponding mlx-c kernel.

func MatMul(a, b *Array, s *Stream) (*Array, error)               { return nil, ErrMLXUnavailable }
func Add(a, b *Array, s *Stream) (*Array, error)                  { return nil, ErrMLXUnavailable }
func Mul(a, b *Array, s *Stream) (*Array, error)                  { return nil, ErrMLXUnavailable }
func Sub(a, b *Array, s *Stream) (*Array, error)                  { return nil, ErrMLXUnavailable }
func Div(a, b *Array, s *Stream) (*Array, error)                  { return nil, ErrMLXUnavailable }
func Softmax(a *Array, axis int, s *Stream) (*Array, error)       { return nil, ErrMLXUnavailable }
func Sigmoid(a *Array, s *Stream) (*Array, error)                 { return nil, ErrMLXUnavailable }
func Tanh(a *Array, s *Stream) (*Array, error)                    { return nil, ErrMLXUnavailable }
func RMSNorm(x, w *Array, eps float32, s *Stream) (*Array, error) { return nil, ErrMLXUnavailable }
func Reshape(a *Array, shape []int, s *Stream) (*Array, error)    { return nil, ErrMLXUnavailable }
func Transpose(a *Array, axes []int, s *Stream) (*Array, error)   { return nil, ErrMLXUnavailable }
func Concatenate(arrs []*Array, axis int, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}
func Split(a *Array, parts, axis int, s *Stream) ([]*Array, error) {
	return nil, ErrMLXUnavailable
}

// SplitSections divides a along axis at the given boundary indices, yielding
// len(indices)+1 sections (the fused-QKV split passes the two query/key
// boundaries to carve out unequal query, key, and value slices).
func SplitSections(a *Array, indices []int, axis int, s *Stream) ([]*Array, error) {
	return nil, ErrMLXUnavailable
}
func Take(a, indices *Array, axis int, s *Stream) (*Array, error) { return nil, ErrMLXUnavailable }
func Argmax(a *Array, axis int, s *Stream) (*Array, error)        { return nil, ErrMLXUnavailable }

// Where selects elementwise from x where cond is true, else from y, broadcasting
// the three operands together. It is the masking and gating primitive the
// deferred forwards need: a sliding-window attention mask and a mixture-of-experts
// router both build their output by selecting between two tensors on a boolean
// condition.
func Where(cond, x, y *Array, s *Stream) (*Array, error) { return nil, ErrMLXUnavailable }

// Cumsum is the cumulative sum of a along axis. reverse accumulates from the end,
// inclusive includes each position's own value in its running total. It is the
// scan primitive the gated-delta recurrence (Qwen3-Next) and the router
// normalization paths build on.
func Cumsum(a *Array, axis int, reverse, inclusive bool, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}

// TakeAlongAxis gathers from a along axis using per-position indices of the same
// rank as a. It is the expert-dispatch gather a mixture-of-experts router uses to
// pull each token's selected expert rows after the top-k pick.
func TakeAlongAxis(a, indices *Array, axis int, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}

// Argpartition returns the indices that place the kth element of a along axis in
// its sorted position, with all smaller-ranked indices before it. A mixture-of-experts
// router takes the top-k expert indices from its partitioned scores this way.
func Argpartition(a *Array, kth, axis int, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}

// GatherMM is the batched gather-matmul a mixture-of-experts block uses to run a
// per-token expert: it multiplies each row of a by the b matrix that rhsIndices
// selects for that row. lhsIndices may be nil (no left gather, the common MoE
// case). When sorted is true the caller has pre-sorted the rows by expert so each
// expert's weight is read once contiguously; the result is identical either way,
// sorted is only a memory-access hint.
func GatherMM(a, b, lhsIndices, rhsIndices *Array, sorted bool, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}

// Argsort returns the indices that sort a along axis. A mixture-of-experts block
// sorts its flattened routing indices this way to group every token bound for the
// same expert, then unsorts the expert outputs with a second Argsort of the order.
func Argsort(a *Array, axis int, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}

// FloorDivide computes element-wise floor(a / b) with broadcasting. The expert
// sort path uses it to map each sorted routing slot back to its token row
// (order // top_k) before the gather.
func FloorDivide(a, b *Array, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}

// Sum reduces a along axis, keeping that dimension as size 1 when keepDims is set.
// The mixture-of-experts router sums each group's two best scores, the
// normalization denominator over the selected experts, and the weighted expert
// outputs along the expert axis with this reduction.
func Sum(a *Array, axis int, keepDims bool, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}

// PutAlongAxis scatters values into a at the positions indices names along axis
// (the inverse of TakeAlongAxis). The group-limited router uses it to zero the
// scores of the expert groups it did not keep before the final top-k pick.
func PutAlongAxis(a, indices, values *Array, axis int, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}

// RoPE applies rotary position embedding with a single base frequency.
func RoPE(x *Array, dims int, traditional bool, base float32, scale float32, offset int, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}

// RoPEWithFreqs applies rotary position embedding from an explicit per-dimension
// frequency table instead of a single base. It is the scaled-rope path: partial
// rotary (Gemma3 proportional, Phi SuScaledRoPE) and the long-context schemes
// (llama3, yarn) all precompute a frequency vector on the host and pass it here.
// A frequency of +Inf leaves that dimension pair unrotated, which is how partial
// rotary rotates only the leading dimensions.
func RoPEWithFreqs(x *Array, dims int, traditional bool, scale float32, offset int, freqs *Array, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}

// ScaledDotProductAttention is the fused attention kernel
// (mlx_fast_scaled_dot_product_attention). maskMode selects a built-in mask
// ("causal" or "" for none); an explicit additive mask array may be passed
// instead.
func ScaledDotProductAttention(q, k, v *Array, scale float32, maskMode string, mask *Array, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}

// QuantizedMatMul multiplies x by a quantized weight (with its scales/biases).
func QuantizedMatMul(x, w, scales, biases *Array, transpose bool, groupSize, bits int, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}

// GatherQMM is the quantized gather-matmul a mixture-of-experts block uses when
// its stacked expert weights are affine-quantized: it multiplies each row of x by
// the quantized expert matrix (w with its scales/biases) that rhsIndices selects,
// the quantized twin of GatherMM. lhsIndices and biases may be nil (no left gather,
// and the bias-free quant modes). transpose, groupSize and bits describe the
// packing; sorted tells the kernel the rows are pre-grouped by expert. The pinned
// mlx-c mlx_gather_qmm has no mode argument, so this is the affine path the routed
// int4 DeepSeek experts use; a later mxfp4 mode awaits an mlx-c bump.
func GatherQMM(x, w, scales, biases, lhsIndices, rhsIndices *Array, transpose bool, groupSize, bits int, sorted bool, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}

// Exp is the elementwise natural exponential. The Qwen3-Next gated-delta
// recurrence builds its decay gate from it: compute_g is the exp of a negated,
// softplus-shaped term.
func Exp(a *Array, s *Stream) (*Array, error) { return nil, ErrMLXUnavailable }

// Logaddexp is the elementwise log(exp(a)+exp(b)). It is the numerically stable
// form of softplus the gated-delta gate needs, since softplus(x) is exactly
// logaddexp(x, 0).
func Logaddexp(a, b *Array, s *Stream) (*Array, error) { return nil, ErrMLXUnavailable }

// Repeat tiles a along axis, repeating each slice `repeats` times in place. The
// gated-delta recurrence repeats the key heads up to the value-head count when a
// layer has more value heads than key heads.
func Repeat(a *Array, repeats, axis int, s *Stream) (*Array, error) {
	return nil, ErrMLXUnavailable
}

func cloneShape(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	out := make([]int, len(shape))
	copy(out, shape)
	return out
}
