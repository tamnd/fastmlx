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
// masked for the rest of the sequence; the same builder serves the decode step with
// L == 1, where the single query sits past every cached key so only the padding skip
// survives the causal term.
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
