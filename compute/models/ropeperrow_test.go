// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"errors"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// ropePerRow is the path a left-padded ragged cohort takes when each row rotates
// at its own offset. With a single row it applies the scalar rope directly (no
// split), so the closure sees the row's offset and the array passes through
// untouched, which is what keeps a lone or uniform cohort on the single-launch
// rope.
func TestRopePerRowSingleRowAppliesScalar(t *testing.T) {
	arr, err := mlxgo.NewFloat32([]float32{1, 2, 3, 4}, 1, 1, 1, 4)
	if err != nil {
		t.Fatalf("NewFloat32: %v", err)
	}
	b := &fb{s: mlxgo.DefaultStream()}
	gotOffset := -1
	calls := 0
	out := b.ropePerRow(arr, []int{7}, func(r *mlxgo.Array, o int) *mlxgo.Array {
		calls++
		gotOffset = o
		return r
	})
	if b.err != nil {
		t.Fatalf("single-row ropePerRow errored: %v", b.err)
	}
	if calls != 1 {
		t.Fatalf("apply called %d times, want 1", calls)
	}
	if gotOffset != 7 {
		t.Fatalf("apply saw offset %d, want 7", gotOffset)
	}
	if out != arr {
		t.Fatalf("single-row ropePerRow did not pass the array through")
	}
}

// With more than one row ropePerRow splits the batch on axis 0 so each row can
// rotate at its own offset; the split is a kernel, so on the default stub the
// builder records the unavailable backend, the same wiring confirmation the rest
// of the forward gives. The split fires before any per-row rope, so apply never
// runs on the stub.
func TestRopePerRowMultiRowReachesSplitSeam(t *testing.T) {
	arr, err := mlxgo.NewFloat32(make([]float32, 2*1*1*4), 2, 1, 1, 4)
	if err != nil {
		t.Fatalf("NewFloat32: %v", err)
	}
	b := &fb{s: mlxgo.DefaultStream()}
	calls := 0
	b.ropePerRow(arr, []int{0, 1}, func(r *mlxgo.Array, o int) *mlxgo.Array {
		calls++
		return r
	})
	if !errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("multi-row ropePerRow err = %v, want ErrMLXUnavailable", b.err)
	}
	if calls != 0 {
		t.Fatalf("apply ran %d times before the split seam, want 0", calls)
	}
}

// A sticky prior error short-circuits ropePerRow: it neither splits nor applies,
// matching every other fb builder method.
func TestRopePerRowShortCircuitsOnPriorError(t *testing.T) {
	arr, err := mlxgo.NewFloat32([]float32{1, 2}, 1, 1, 1, 2)
	if err != nil {
		t.Fatalf("NewFloat32: %v", err)
	}
	b := &fb{s: mlxgo.DefaultStream(), err: errors.New("prior")}
	calls := 0
	out := b.ropePerRow(arr, []int{0, 1}, func(r *mlxgo.Array, o int) *mlxgo.Array {
		calls++
		return r
	})
	if out != nil {
		t.Fatalf("ropePerRow returned non-nil after a prior error")
	}
	if calls != 0 {
		t.Fatalf("apply ran %d times after a prior error, want 0", calls)
	}
}
