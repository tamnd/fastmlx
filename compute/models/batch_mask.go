// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import "github.com/tamnd/fastmlx/mlxgo"

// batchLeftPadCausalData builds the per-row additive prefill mask for a left-padded
// batch, returned as a flat row-major [batch, 1, L, offset+L] float32 buffer. It is
// the faithful port of the reference create_causal_mask called with a left_padding
// argument: for row b, query position offset+i attends to key position r only when
// r is at or before the query (causal) and at or after that row's left padding, so
// the padding tokens prepended to a short prompt are skipped. leftPad[b] is the
// number of padding tokens prepended to row b (the merged length minus that prompt's
// true length). Attendable entries are zero and the rest a large negative, the same
// additive convention causalAdditiveMask and slidingWindowMask use, so the mask
// feeds the existing explicit-mask SDPA path with no new kernel.
//
// Left padding is what lets a ragged cohort decode in lockstep: every row's last
// real token lands at the same position after prefill, so the synchronized [batch,
// 1] decode that follows shares one offset. The trade is that the padding keys stay
// masked for the rest of the sequence, which a decode-time companion mask carries.
func batchLeftPadCausalData(leftPad []int, L, offset int) []float32 {
	S := offset + L
	batch := len(leftPad)
	m := make([]float32, batch*L*S)
	const negInf = -1e30
	for b := range batch {
		pad := leftPad[b]
		base := b * L * S
		for i := range L {
			qpos := offset + i
			row := base + i*S
			for j := range S {
				if j > qpos || j < pad {
					m[row+j] = negInf
				}
			}
		}
	}
	return m
}

// batchLeftPadCausalMask wraps batchLeftPadCausalData in a [batch, 1, L, offset+L]
// array ready for the explicit-mask SDPA. The single head axis broadcasts the mask
// across every attention head. The array build is host-side, so it materializes on
// the default stub too; only feeding it to SDPA engages a kernel.
func batchLeftPadCausalMask(leftPad []int, L, offset int, s *mlxgo.Stream) (*mlxgo.Array, error) {
	S := offset + L
	return mlxgo.NewFloat32(batchLeftPadCausalData(leftPad, L, offset), len(leftPad), 1, L, S)
}

// batchLeftPadKeyData builds the per-row additive decode mask for a left-padded
// batch, a flat row-major [batch, 1, 1, offset] float32 buffer. After the prefill
// every row decodes one token at a time against the grown cache, and the padding
// keys at the front must stay masked: key position r is attendable only when r is at
// or after that row's left padding. This is the create_causal_mask left_padding
// term specialized to the single-query decode step (the causal term is vacuous when
// the one query sits past every cached key), so a synchronized but left-padded
// cohort still needs a mask where an unpadded one needs none.
func batchLeftPadKeyData(leftPad []int, offset int) []float32 {
	batch := len(leftPad)
	m := make([]float32, batch*offset)
	const negInf = -1e30
	for b := range batch {
		pad := leftPad[b]
		base := b * offset
		for r := range offset {
			if r < pad {
				m[base+r] = negInf
			}
		}
	}
	return m
}

// batchLeftPadKeyMask wraps batchLeftPadKeyData in a [batch, 1, 1, offset] array for
// the explicit-mask SDPA of a left-padded decode step. A cohort with no padding
// (every leftPad zero) produces an all-zero mask, which the caller may skip; the
// builder still returns it so the shape is always available.
func batchLeftPadKeyMask(leftPad []int, offset int, s *mlxgo.Stream) (*mlxgo.Array, error) {
	return mlxgo.NewFloat32(batchLeftPadKeyData(leftPad, offset), len(leftPad), 1, 1, offset)
}
