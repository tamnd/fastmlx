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
	want := []int{3, 1, 1, 5} // [batch, 1, 1, offset]
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
	eqFloats(t, data, batchLeftPadKeyData(leftPad, 5), "left-padded decode mask data")
}
