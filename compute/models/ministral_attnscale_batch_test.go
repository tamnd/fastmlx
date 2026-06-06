// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import "testing"

// AttnScaleBatch is the per-row llama4 query scale a left-padded cohort needs.
// It must equal the single-row AttnScale evaluated at each row's offset, laid out
// row-major, since a ragged cohort's rows sit at different logical positions. A
// beta and a small max-position are chosen so the floor term actually steps and
// the log contributes, rather than the all-ones degenerate scale.
func TestAttnScaleBatchMatchesPerRow(t *testing.T) {
	a := &MinistralArgs{Llama4ScalingBeta: 0.5, OriginalMaxPositionEmbeddings: 4}
	offsets := []int{0, 3, 6}
	const L = 5
	got := a.AttnScaleBatch(L, offsets)
	if len(got) != len(offsets)*L {
		t.Fatalf("AttnScaleBatch len %d, want %d", len(got), len(offsets)*L)
	}
	for b, off := range offsets {
		want := a.AttnScale(L, off)
		for i := range want {
			if got[b*L+i] != want[i] {
				t.Fatalf("row %d pos %d = %g, want %g", b, i, got[b*L+i], want[i])
			}
		}
	}
}

// Equal offsets across rows reproduce the single-row scale in every row, the
// degenerate case that pins the per-row build as a strict generalization of the
// uniform [1, 1, L, 1] scale.
func TestAttnScaleBatchUniformOffsetsMatchSingle(t *testing.T) {
	a := &MinistralArgs{Llama4ScalingBeta: 0.25, OriginalMaxPositionEmbeddings: 8}
	const L = 4
	single := a.AttnScale(L, 2)
	batch := a.AttnScaleBatch(L, []int{2, 2})
	for b := range 2 {
		for i := range single {
			if batch[b*L+i] != single[i] {
				t.Fatalf("row %d pos %d = %g, want %g", b, i, batch[b*L+i], single[i])
			}
		}
	}
}
