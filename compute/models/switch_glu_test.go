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

// switchQuantWeights fabricates the three affine-quantized stacked-expert tensors
// for a SwitchGLU with numExperts experts, hidden width d and expert width h, at
// group_size G and int4 packing. The packed weight folds 8 int4 values per uint32,
// and the scales/biases carry one entry per group along the input axis. Only the
// shapes matter on the stub; gather_qmm is the first kernel and never runs.
func switchQuantWeights(t *testing.T, numExperts, d, h, group int) (gate, up, down switchQuant) {
	t.Helper()
	const bits = 4
	mkF := func(shape ...int) *mlxgo.Array {
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
	// The packed weight is uint32 on the device; mlxgo exposes no uint32 builder and
	// the stub holds it as opaque host data, so an int32 array of the same shape
	// stands in. gather_qmm is the first kernel and never runs.
	mkU := func(shape ...int) *mlxgo.Array {
		n := 1
		for _, s := range shape {
			n *= s
		}
		a, err := mlxgo.NewInt32(make([]int32, n), shape...)
		if err != nil {
			t.Fatalf("NewInt32%v: %v", shape, err)
		}
		return a
	}
	// gate_proj/up_proj map d->h (in=d), down_proj maps h->d (in=h). Stored
	// [experts, out, in]; packed weight has in*bits/32 words per row, scales and
	// biases one entry per in/group group.
	q := func(in, out int) switchQuant {
		return switchQuant{
			w:         mkU(numExperts, out, in*bits/32),
			scales:    mkF(numExperts, out, in/group),
			biases:    mkF(numExperts, out, in/group),
			groupSize: group,
			bits:      bits,
		}
	}
	return q(d, h), q(d, h), q(h, d)
}

// TestSwitchGLUQuantizedGracefulNoSort drives the short routing through the
// quantized projections: the host shape choreography runs and the forward reports
// the missing backend at the first gather_qmm rather than panicking.
func TestSwitchGLUQuantizedGracefulNoSort(t *testing.T) {
	gate, up, down := switchQuantWeights(t, 5, 64, 128, 64)
	x, inds := switchInputs(t, 2, 64, 3) // 2*3 = 6 slots < 64, no sort
	b := &fb{s: mlxgo.DefaultStream()}
	if got := b.switchGLUQuantized(x, gate, up, down, inds); got != nil {
		t.Fatalf("switchGLUQuantized result = %v, want nil on the stub", got)
	}
	if !errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
		t.Errorf("switchGLUQuantized err = %v, want ErrMLXUnavailable", b.err)
	}
}

// TestSwitchGLUQuantizedGracefulSort drives the long routing (sort path) through
// the quantized projections.
func TestSwitchGLUQuantizedGracefulSort(t *testing.T) {
	gate, up, down := switchQuantWeights(t, 8, 64, 128, 64)
	x, inds := switchInputs(t, 32, 64, 2) // 32*2 = 64 slots, sort path
	b := &fb{s: mlxgo.DefaultStream()}
	if got := b.switchGLUQuantized(x, gate, up, down, inds); got != nil {
		t.Fatalf("switchGLUQuantized result = %v, want nil on the stub", got)
	}
	if !errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
		t.Errorf("switchGLUQuantized err = %v, want ErrMLXUnavailable", b.err)
	}
}

// TestSwitchGLUQuantizedBadShape rejects a non-rank-3 hidden state before touching
// the backend, sharing the dense path's guard.
func TestSwitchGLUQuantizedBadShape(t *testing.T) {
	gate, up, down := switchQuantWeights(t, 5, 64, 128, 64)
	x, err := mlxgo.NewFloat32(make([]float32, 128), 2, 64) // rank 2, not [B,L,D]
	if err != nil {
		t.Fatalf("NewFloat32: %v", err)
	}
	inds, err := mlxgo.NewInt32(make([]int32, 6), 1, 2, 3)
	if err != nil {
		t.Fatalf("NewInt32: %v", err)
	}
	b := &fb{s: mlxgo.DefaultStream()}
	b.switchGLUQuantized(x, gate, up, down, inds)
	if b.err == nil || errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
		t.Errorf("err = %v, want a shape error before any kernel", b.err)
	}
}

// TestSwitchGLUQuantizedStickyError confirms a pre-existing builder error
// short-circuits the quantized block too.
func TestSwitchGLUQuantizedStickyError(t *testing.T) {
	gate, up, down := switchQuantWeights(t, 5, 64, 128, 64)
	x, inds := switchInputs(t, 2, 64, 3)
	sentinel := errors.New("prior failure")
	b := &fb{s: mlxgo.DefaultStream(), err: sentinel}
	if got := b.switchGLUQuantized(x, gate, up, down, inds); got != nil {
		t.Fatalf("result = %v, want nil", got)
	}
	if b.err != sentinel {
		t.Errorf("err = %v, want the original sentinel", b.err)
	}
}

func BenchmarkSwitchGLUQuantizedNoSort(b *testing.B) {
	gate, up, down := switchQuantWeights(&testing.T{}, 5, 64, 128, 64)
	x, _ := mlxgo.NewFloat32(make([]float32, 128), 1, 2, 64)
	inds, _ := mlxgo.NewInt32(make([]int32, 6), 1, 2, 3)
	s := mlxgo.DefaultStream()
	b.ReportAllocs()
	for b.Loop() {
		fbb := &fb{s: s}
		fbb.switchGLUQuantized(x, gate, up, down, inds)
	}
}
