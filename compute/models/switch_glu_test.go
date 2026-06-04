// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"errors"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// switchWeights fabricates the three stacked expert tensors for a SwitchGLU with
// numExperts experts, hidden width d and expert intermediate width h. Shapes are
// what matter; the stub holds them as opaque host data.
func switchWeights(t *testing.T, numExperts, d, h int) (gate, up, down *mlxgo.Array) {
	t.Helper()
	mk := func(shape ...int) *mlxgo.Array {
		n := 1
		for _, s := range shape {
			n *= s
		}
		a, err := mlxgo.NewFloat32(make([]float32, n), shape...)
		if err != nil {
			t.Fatalf("NewFloat32%v: %v", shape, err)
		}
		return a
	}
	// gate_proj/up_proj map d->h, down_proj maps h->d; stored [experts, out, in].
	return mk(numExperts, h, d), mk(numExperts, h, d), mk(numExperts, d, h)
}

func switchInputs(t *testing.T, seqLen, d, topK int) (x, inds *mlxgo.Array) {
	t.Helper()
	x, err := mlxgo.NewFloat32(make([]float32, seqLen*d), 1, seqLen, d)
	if err != nil {
		t.Fatalf("NewFloat32 x: %v", err)
	}
	idx := make([]int32, seqLen*topK)
	inds, err = mlxgo.NewInt32(idx, 1, seqLen, topK)
	if err != nil {
		t.Fatalf("NewInt32 inds: %v", err)
	}
	return x, inds
}

// TestSwitchGLUGracefulNoSort drives the short routing (fewer than 64 slots), which
// skips the expert sort and gather-matmuls directly. The forward type-checks and
// runs the host-side shape choreography, then reports the missing backend at the
// first kernel instead of panicking.
func TestSwitchGLUGracefulNoSort(t *testing.T) {
	gate, up, down := switchWeights(t, 5, 4, 6)
	x, inds := switchInputs(t, 2, 4, 3) // 2*3 = 6 slots < 64, no sort
	b := &fb{s: mlxgo.DefaultStream()}
	if got := b.switchGLU(x, gate, up, down, inds); got != nil {
		t.Fatalf("switchGLU result = %v, want nil on the stub", got)
	}
	if !errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
		t.Errorf("switchGLU err = %v, want ErrMLXUnavailable", b.err)
	}
}

// TestSwitchGLUGracefulSort drives the long routing (at least 64 slots), which
// takes the sort/gather/unsort path. It must also run its host shape math and
// report the missing backend rather than panicking on a shape mismatch.
func TestSwitchGLUGracefulSort(t *testing.T) {
	gate, up, down := switchWeights(t, 8, 4, 6)
	x, inds := switchInputs(t, 32, 4, 2) // 32*2 = 64 slots, sort path
	b := &fb{s: mlxgo.DefaultStream()}
	if got := b.switchGLU(x, gate, up, down, inds); got != nil {
		t.Fatalf("switchGLU result = %v, want nil on the stub", got)
	}
	if !errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
		t.Errorf("switchGLU err = %v, want ErrMLXUnavailable", b.err)
	}
}

// TestSwitchGLUBadShape rejects a hidden state or routing selection that is not
// rank 3 before touching the backend, with a clear error.
func TestSwitchGLUBadShape(t *testing.T) {
	gate, up, down := switchWeights(t, 5, 4, 6)
	x, err := mlxgo.NewFloat32(make([]float32, 8), 2, 4) // rank 2, not [B,L,D]
	if err != nil {
		t.Fatalf("NewFloat32: %v", err)
	}
	inds, err := mlxgo.NewInt32(make([]int32, 6), 1, 2, 3)
	if err != nil {
		t.Fatalf("NewInt32: %v", err)
	}
	b := &fb{s: mlxgo.DefaultStream()}
	b.switchGLU(x, gate, up, down, inds)
	if b.err == nil || errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
		t.Errorf("err = %v, want a shape error before any kernel", b.err)
	}
}

// TestSwitchGLUStickyError confirms a pre-existing builder error short-circuits the
// block: it returns nil and leaves the original error untouched.
func TestSwitchGLUStickyError(t *testing.T) {
	gate, up, down := switchWeights(t, 5, 4, 6)
	x, inds := switchInputs(t, 2, 4, 3)
	sentinel := errors.New("prior failure")
	b := &fb{s: mlxgo.DefaultStream(), err: sentinel}
	if got := b.switchGLU(x, gate, up, down, inds); got != nil {
		t.Fatalf("result = %v, want nil", got)
	}
	if b.err != sentinel {
		t.Errorf("err = %v, want the original sentinel", b.err)
	}
}

func BenchmarkSwitchGLUNoSort(b *testing.B) {
	gate, up, down := switchWeights(&testing.T{}, 5, 4, 6)
	x, _ := mlxgo.NewFloat32(make([]float32, 8), 1, 2, 4)
	inds, _ := mlxgo.NewInt32(make([]int32, 6), 1, 2, 3)
	s := mlxgo.DefaultStream()
	b.ReportAllocs()
	for b.Loop() {
		fbb := &fb{s: s}
		fbb.switchGLU(x, gate, up, down, inds)
	}
}
