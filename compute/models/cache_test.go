// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// SetLeftPad normalizes an all-zero (or nil) cohort to nil so the uniform fast
// paths stay engaged, and keeps a genuinely ragged slice verbatim.
func TestSetLeftPadNormalizesUniform(t *testing.T) {
	cases := []struct {
		name string
		in   []int
		want []int
	}{
		{"nil", nil, nil},
		{"all zero", []int{0, 0, 0}, nil},
		{"ragged", []int{0, 2, 1}, []int{0, 2, 1}},
		{"single padded", []int{3}, []int{3}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := &KVTensorCache{}
			c.SetLeftPad(tc.in)
			got := c.LeftPad()
			if len(got) != len(tc.want) {
				t.Fatalf("LeftPad() = %v, want %v", got, tc.want)
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Fatalf("LeftPad() = %v, want %v", got, tc.want)
				}
			}
		})
	}
}

// RopeOffsets is nil for a uniform cohort (the scalar-offset RoPE fast path) and
// otherwise Offset-leftPad[b] per row, the reference's per-row cache offset of
// -leftPad advanced by the tokens seen.
func TestRopeOffsets(t *testing.T) {
	uniform := &KVTensorCache{Offset: 7}
	if off := uniform.RopeOffsets(); off != nil {
		t.Fatalf("uniform RopeOffsets() = %v, want nil", off)
	}

	ragged := &KVTensorCache{Offset: 6}
	ragged.SetLeftPad([]int{0, 2, 5})
	want := []int{6, 4, 1}
	got := ragged.RopeOffsets()
	if len(got) != len(want) {
		t.Fatalf("RopeOffsets() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("RopeOffsets() = %v, want %v", got, want)
		}
	}
}

// A uniform cohort reproduces the hardcoded mask behavior with no explicit mask:
// "causal" for a multi-token prefill, "" for a single-token decode step.
func TestAttnMaskUniform(t *testing.T) {
	c := &KVTensorCache{Offset: 4}
	mode, mask, err := c.AttnMask(2, 3, mlxgo.DefaultStream())
	if err != nil {
		t.Fatalf("prefill AttnMask: %v", err)
	}
	if mode != "causal" || mask != nil {
		t.Fatalf("uniform prefill = (%q, %v), want (\"causal\", nil)", mode, mask)
	}
	mode, mask, err = c.AttnMask(2, 1, mlxgo.DefaultStream())
	if err != nil {
		t.Fatalf("decode AttnMask: %v", err)
	}
	if mode != "" || mask != nil {
		t.Fatalf("uniform decode = (%q, %v), want (\"\", nil)", mode, mask)
	}
}

// A left-padded cohort returns an empty mode and an explicit additive mask. The
// mask is built host-side, so it materializes on the default stub; its shape and
// contents match the standalone batch_mask builders the helper draws on.
func TestAttnMaskLeftPaddedPrefill(t *testing.T) {
	leftPad := []int{0, 2}
	c := &KVTensorCache{Offset: 0}
	c.SetLeftPad(leftPad)
	mode, mask, err := c.AttnMask(2, 3, mlxgo.DefaultStream())
	if err != nil {
		t.Fatalf("AttnMask: %v", err)
	}
	if mode != "" {
		t.Fatalf("left-padded prefill mode = %q, want \"\"", mode)
	}
	want := []int{2, 1, 3, 3} // [batch, 1, L, offset+L]
	got := mask.Shape()
	if len(got) != len(want) {
		t.Fatalf("mask shape %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("mask shape %v, want %v", got, want)
		}
	}
	data, err := mask.ToFloat32()
	if err != nil {
		t.Fatalf("mask Float32s: %v", err)
	}
	eqFloats(t, data, batchLeftPadCausalData(leftPad, 3, 0), "left-padded prefill mask data")
}

func TestAttnMaskLeftPaddedDecode(t *testing.T) {
	leftPad := []int{0, 3, 1}
	c := &KVTensorCache{Offset: 5}
	c.SetLeftPad(leftPad)
	mode, mask, err := c.AttnMask(3, 1, mlxgo.DefaultStream())
	if err != nil {
		t.Fatalf("AttnMask: %v", err)
	}
	if mode != "" {
		t.Fatalf("left-padded decode mode = %q, want \"\"", mode)
	}
	want := []int{3, 1, 1, 6} // [batch, 1, 1, offset+L], the post-update key length
	got := mask.Shape()
	if len(got) != len(want) {
		t.Fatalf("mask shape %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("mask shape %v, want %v", got, want)
		}
	}
	data, err := mask.ToFloat32()
	if err != nil {
		t.Fatalf("mask Float32s: %v", err)
	}
	eqFloats(t, data, batchLeftPadCausalData(leftPad, 1, 5), "left-padded decode mask data")
}

// mergeCachesAlongBatch reassembles each sequence's own one-element leftPad into
// the merged per-row slice and splitCachesAlongBatch writes each row's element
// back, so a cohort prefilled together keeps masking its front padding across
// steps. Fresh (nil-tensor) caches exercise the bookkeeping without a kernel.
func TestMergeSplitCarriesLeftPad(t *testing.T) {
	// Two layers, three sequences; seq 1 and 2 carry front padding, seq 0 none.
	seqs := [][]*KVTensorCache{
		{{Offset: 5}, {Offset: 5}},
		{{Offset: 5}, {Offset: 5}},
		{{Offset: 5}, {Offset: 5}},
	}
	seqs[1][0].SetLeftPad([]int{2})
	seqs[1][1].SetLeftPad([]int{2})
	seqs[2][0].SetLeftPad([]int{1})
	seqs[2][1].SetLeftPad([]int{1})

	merged, err := mergeCachesAlongBatch(seqs, nil)
	if err != nil {
		t.Fatalf("merge: %v", err)
	}
	for l, mc := range merged {
		if want := []int{0, 2, 1}; !eqInts(mc.LeftPad(), want) {
			t.Fatalf("layer %d merged leftPad = %v, want %v", l, mc.LeftPad(), want)
		}
	}

	// Grow the offset and wipe each sequence's leftPad, so the split has to rewrite
	// both from the merged cache.
	for _, mc := range merged {
		mc.Offset = 6
	}
	for _, seq := range seqs {
		for _, c := range seq {
			c.SetLeftPad(nil)
		}
	}
	if err := splitCachesAlongBatch(merged, seqs, nil); err != nil {
		t.Fatalf("split: %v", err)
	}
	wantPad := []int{0, 2, 1}
	for i, seq := range seqs {
		for l, c := range seq {
			if c.Offset != 6 {
				t.Fatalf("seq %d layer %d offset = %d, want 6", i, l, c.Offset)
			}
			lp := c.LeftPad()
			if wantPad[i] == 0 {
				if lp != nil {
					t.Fatalf("seq %d layer %d leftPad = %v, want nil", i, l, lp)
				}
			} else if len(lp) != 1 || lp[0] != wantPad[i] {
				t.Fatalf("seq %d layer %d leftPad = %v, want [%d]", i, l, lp, wantPad[i])
			}
		}
	}
}

// A cohort with no padding merges to a nil leftPad (the uniform fast path), and
// the split leaves every sequence's leftPad nil, so a normally-prefilled batched
// decode is byte for byte unchanged by the carry.
func TestMergeSplitNoPaddingStaysUniform(t *testing.T) {
	seqs := [][]*KVTensorCache{{{Offset: 4}}, {{Offset: 4}}}
	merged, err := mergeCachesAlongBatch(seqs, nil)
	if err != nil {
		t.Fatalf("merge: %v", err)
	}
	if merged[0].LeftPad() != nil {
		t.Fatalf("no-padding cohort merged leftPad = %v, want nil", merged[0].LeftPad())
	}
	if err := splitCachesAlongBatch(merged, seqs, nil); err != nil {
		t.Fatalf("split: %v", err)
	}
	for i, seq := range seqs {
		if seq[0].LeftPad() != nil {
			t.Fatalf("seq %d leftPad = %v, want nil", i, seq[0].LeftPad())
		}
	}
}
