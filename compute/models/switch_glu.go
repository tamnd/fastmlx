// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"fmt"

	"github.com/tamnd/fastmlx/mlxgo"
)

// errBadSwitchShape reports a switchGLU input that is not the expected rank-3
// hidden state and rank-3 routing selection.
func errBadSwitchShape(xShape, indShape []int) error {
	return fmt.Errorf("switchGLU: want x [B,L,D] and inds [B,L,top_k], got x %v inds %v", xShape, indShape)
}

// switchGLU is the mixture-of-experts expert MLP, the SwitchGLU block every routed
// family (DeepSeek-V3, Qwen3-Next, Gemma4 MoE) runs after its router picks the
// per-token experts. x is the per-token hidden state [B, L, D]; inds is the
// per-token expert selection [B, L, top_k]. The three weights are the stacked
// expert tensors gate_proj/up_proj/down_proj, each shaped [num_experts, out, in].
// The result is [B, L, top_k, D]: one expert MLP output per selected expert, which
// the caller weights by the router scores and sums over the top_k axis.
//
// The shape choreography mirrors the reference SwitchGLU exactly. x gains two
// singleton dims (expand_dims at -2 and -3) so each token broadcasts against its
// top_k experts inside the gather-matmul; the trailing singleton matmul-row dim is
// dropped at the end (squeeze at -2). When the routing fans out to at least 64
// slots the rows are sorted by expert first, so each expert's weight is read once
// contiguously, then unsorted afterward; the result is identical either way, the
// sort is only a memory-access win. expand_dims, squeeze, flatten and unflatten
// are all layout-preserving, so they are plain reshapes over host-known shapes.
func (b *fb) switchGLU(x, gateW, upW, downW, inds *mlxgo.Array) *mlxgo.Array {
	return b.switchGLUCore(x, inds, func(xe, idx *mlxgo.Array, which gluProj, sorted bool) *mlxgo.Array {
		switch which {
		case projGate:
			return b.switchLinear(xe, gateW, idx, sorted)
		case projUp:
			return b.switchLinear(xe, upW, idx, sorted)
		default:
			return b.switchLinear(xe, downW, idx, sorted)
		}
	})
}

// switchQuant is one affine-quantized stacked-expert projection: the packed
// weight, its per-group scales and biases, and the group_size/bits that describe
// the packing. It is the quantized twin of the dense weight switchGLU consumes,
// the QuantizedSwitchLinear the reference swaps in when a checkpoint stores its
// experts int4.
type switchQuant struct {
	w, scales, biases *mlxgo.Array
	groupSize, bits   int
}

// switchGLUQuantized is switchGLU over affine-quantized experts: identical routing
// choreography, but each projection is a quantized gather-matmul (gather_qmm)
// against the packed weight and its scales/biases instead of a dense gather-matmul.
// The routed experts carry no bias, matching QuantizedSwitchLinear(bias=False).
func (b *fb) switchGLUQuantized(x *mlxgo.Array, gate, up, down switchQuant, inds *mlxgo.Array) *mlxgo.Array {
	return b.switchGLUCore(x, inds, func(xe, idx *mlxgo.Array, which gluProj, sorted bool) *mlxgo.Array {
		q := gate
		switch which {
		case projUp:
			q = up
		case projDown:
			q = down
		}
		return b.switchLinearQuantized(xe, q, idx, sorted)
	})
}

// gluProj names the three stacked-expert projections so switchGLUCore can ask the
// projection callback for the right weight without threading three arrays through.
type gluProj int

const (
	projGate gluProj = iota
	projUp
	projDown
)

// switchGLUCore is the shared SwitchGLU choreography: it runs the sort/gather
// bookkeeping and the silu(gate)*up gating, deferring each per-expert projection
// to proj, which selects the dense or quantized weight for gluProj which and
// applies it to x with the given sorted hint. switchGLU and switchGLUQuantized are
// the two instantiations; the host-side reshape math is identical for both.
func (b *fb) switchGLUCore(x, inds *mlxgo.Array, proj func(x, idx *mlxgo.Array, which gluProj, sorted bool) *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	xs := x.Shape()
	is := inds.Shape()
	if len(xs) != 3 || len(is) != 3 {
		b.err = errBadSwitchShape(xs, is)
		return nil
	}
	bsz, seqLen, dim := xs[0], xs[1], xs[2]
	topK := is[2]
	slots := bsz * seqLen * topK

	// expand_dims(x, (-2, -3)): [B, L, D] -> [B, L, 1, 1, D].
	xe := b.reshape(x, []int{bsz, seqLen, 1, 1, dim})

	if slots < 64 {
		up := proj(xe, inds, projUp, false)
		gate := proj(xe, inds, projGate, false)
		h := b.mul(b.silu(gate), up)
		down := proj(h, inds, projDown, false)
		// squeeze the matmul-row dim: [B, L, top_k, 1, D] -> [B, L, top_k, D].
		return b.reshape(down, []int{bsz, seqLen, topK, dim})
	}

	// Sort the flattened routing slots by expert so the gather reads each expert
	// once, then build the inverse permutation to restore token order.
	indFlat := b.reshape(inds, []int{slots})
	order := b.argsort(indFlat, 0)
	invOrder := b.argsort(order, 0)
	// x.flatten(0, -3): [B, L, 1, 1, D] -> [B*L, 1, D]; gather the token row for
	// each slot (order // top_k maps a sorted slot back to its token).
	xFlat := b.reshape(xe, []int{bsz * seqLen, 1, dim})
	rows := b.floorDivideScalar(order, topK)
	xSorted := b.take(xFlat, rows, 0)
	idx := b.take(indFlat, order, 0)

	up := proj(xSorted, idx, projUp, true)
	gate := proj(xSorted, idx, projGate, true)
	h := b.mul(b.silu(gate), up)
	down := proj(h, idx, projDown, true)

	unsorted := b.take(down, invOrder, 0)
	// unflatten to [B, L, top_k, 1, D] and squeeze: [B, L, top_k, D].
	return b.reshape(unsorted, []int{bsz, seqLen, topK, dim})
}

// switchLinear is one stacked-expert projection: a gather-matmul that multiplies
// each row by the expert matrix its index selects. The weight is stored
// [num_experts, out, in]; swapping the last two axes gives the [num_experts, in,
// out] the gather-matmul multiplies against. sorted tells the kernel the rows are
// already grouped by expert. The routed experts carry no bias.
func (b *fb) switchLinear(x, w, idx *mlxgo.Array, sorted bool) *mlxgo.Array {
	wt := b.transpose(w, []int{0, 2, 1})
	return b.gatherMM(x, wt, idx, sorted)
}

func (b *fb) gatherMM(a, w, rhsIndices *mlxgo.Array, sorted bool) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.GatherMM(a, w, nil, rhsIndices, sorted, b.s)
	b.err = err
	return r
}

// switchLinearQuantized is the quantized twin of switchLinear: one stacked-expert
// projection through gather_qmm. The reference QuantizedSwitchLinear keeps its
// packed weight in [num_experts, out, in] and passes transpose=true to the kernel
// rather than pre-swapping the axes the way the dense switchLinear does, so no
// host transpose runs here. The routed experts carry no bias.
func (b *fb) switchLinearQuantized(x *mlxgo.Array, q switchQuant, idx *mlxgo.Array, sorted bool) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.GatherQMM(x, q.w, q.scales, q.biases, nil, idx, true, q.groupSize, q.bits, sorted, b.s)
	b.err = err
	return r
}

func (b *fb) argsort(x *mlxgo.Array, axis int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Argsort(x, axis, b.s)
	b.err = err
	return r
}

// floorDivideScalar computes x // c for an integer scalar c, the order // top_k
// slot-to-token map the expert sort needs.
func (b *fb) floorDivideScalar(x *mlxgo.Array, c int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	ca, err := mlxgo.NewInt32([]int32{int32(c)}, 1)
	if err != nil {
		b.err = err
		return nil
	}
	r, err := mlxgo.FloorDivide(x, ca, b.s)
	b.err = err
	return r
}
