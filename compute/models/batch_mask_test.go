// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

const maskNegInf = -1e30

// wantCausal recomputes the left-padded causal mask straight from the reference
// predicate (causal AND past the row's left padding), the independent oracle the
// builder is checked against.
func wantCausal(leftPad []int, L, offset int) []float32 {
	S := offset + L
	out := make([]float32, len(leftPad)*L*S)
	for b, pad := range leftPad {
		for i := range L {
			qpos := offset + i
			for j := range S {
				idx := b*L*S + i*S + j
				if j <= qpos && j >= pad {
					out[idx] = 0
				} else {
					out[idx] = maskNegInf
				}
			}
		}
	}
	return out
}

func eqFloats(t *testing.T, got, want []float32, ctx string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", ctx, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: entry %d = %g, want %g", ctx, i, got[i], want[i])
		}
	}
}

func TestBatchLeftPadCausalData(t *testing.T) {
	// Three rows padded by 0, 2, and 1: the unpadded row is plain causal, the
	// others additionally drop their front padding keys.
	leftPad := []int{0, 2, 1}
	L, offset := 4, 0
	got := batchLeftPadCausalData(leftPad, L, offset)
	eqFloats(t, got, wantCausal(leftPad, L, offset), "causal data")
}

func TestBatchLeftPadCausalDataWithOffset(t *testing.T) {
	// A non-zero offset (cached context before this block) shifts the query
	// positions; the builder must compare keys against the global query position.
	leftPad := []int{0, 3}
	L, offset := 2, 5
	got := batchLeftPadCausalData(leftPad, L, offset)
	eqFloats(t, got, wantCausal(leftPad, L, offset), "causal data with offset")
}

// TestBatchLeftPadCausalZeroPaddingMatchesSingle pins the reduction: a row with no
// left padding is byte-for-byte the single-sequence causalAdditiveMask, so the
// batched mask is a strict generalization of the path already serving prefill.
func TestBatchLeftPadCausalZeroPaddingMatchesSingle(t *testing.T) {
	L, offset := 5, 3
	batched := batchLeftPadCausalData([]int{0}, L, offset)
	single := causalAdditiveMask(L, offset)
	eqFloats(t, batched, single, "zero-padding row vs single-sequence mask")
}

func TestBatchLeftPadCausalFullPaddingRow(t *testing.T) {
	// A row padded by the whole length has every key before some query masked by
	// the padding term; only the diagonal-and-past keys that are also past the pad
	// survive. The oracle covers it, so this just guards the extreme.
	leftPad := []int{4}
	L, offset := 4, 0
	got := batchLeftPadCausalData(leftPad, L, offset)
	// Row 0 query 0 (qpos 0) can attend to no key (every key j<4 is below pad 4
	// except none, and j>0 is future), so its whole row is masked.
	for j := range L {
		if got[j] != maskNegInf {
			t.Fatalf("fully padded query 0 key %d = %g, want masked", j, got[j])
		}
	}
	eqFloats(t, got, wantCausal(leftPad, L, offset), "full-padding row")
}

func TestBatchLeftPadKeyData(t *testing.T) {
	// Decode masks only the front padding keys; the causal term is vacuous because
	// the single new query sits past every cached key.
	leftPad := []int{0, 3, 1}
	offset := 5
	got := batchLeftPadKeyData(leftPad, offset)
	for b, pad := range leftPad {
		for r := range offset {
			want := float32(0)
			if r < pad {
				want = maskNegInf
			}
			if got[b*offset+r] != want {
				t.Fatalf("decode mask row %d key %d = %g, want %g", b, r, got[b*offset+r], want)
			}
		}
	}
}

func TestBatchLeftPadKeyDataNoPadding(t *testing.T) {
	// An unpadded cohort produces an all-zero decode mask (the caller may skip it).
	got := batchLeftPadKeyData([]int{0, 0}, 4)
	for i, v := range got {
		if v != 0 {
			t.Fatalf("unpadded decode mask entry %d = %g, want 0", i, v)
		}
	}
}

// The mask wrappers build their arrays host-side, so they succeed on the default
// stub (array creation is not a kernel) and carry the [batch, 1, L, S] and
// [batch, 1, 1, offset] shapes the explicit-mask SDPA expects.
func TestBatchLeftPadCausalMaskShape(t *testing.T) {
	m, err := batchLeftPadCausalMask([]int{0, 2}, 3, 1, mlxgo.DefaultStream())
	if err != nil {
		t.Fatalf("batchLeftPadCausalMask: %v", err)
	}
	want := []int{2, 1, 3, 4} // [batch, 1, L, offset+L]
	got := m.Shape()
	if len(got) != len(want) {
		t.Fatalf("causal mask shape %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("causal mask shape %v, want %v", got, want)
		}
	}
}

func TestBatchLeftPadKeyMaskShape(t *testing.T) {
	m, err := batchLeftPadKeyMask([]int{1, 0, 3}, 6, mlxgo.DefaultStream())
	if err != nil {
		t.Fatalf("batchLeftPadKeyMask: %v", err)
	}
	want := []int{3, 1, 1, 6} // [batch, 1, 1, offset]
	got := m.Shape()
	if len(got) != len(want) {
		t.Fatalf("key mask shape %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("key mask shape %v, want %v", got, want)
		}
	}
}

func BenchmarkBatchLeftPadCausalData(b *testing.B) {
	leftPad := []int{0, 1, 2, 3, 4, 5, 6, 7}
	b.ReportAllocs()
	for b.Loop() {
		_ = batchLeftPadCausalData(leftPad, 64, 0)
	}
}
