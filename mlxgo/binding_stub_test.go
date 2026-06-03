// SPDX-License-Identifier: MIT OR Apache-2.0

//go:build !mlx

package mlxgo

import (
	"errors"
	"reflect"
	"testing"
)

func TestDtypeNameAndSize(t *testing.T) {
	cases := []struct {
		d    Dtype
		name string
		size int
	}{
		{Bool, "bool", 1},
		{Uint8, "uint8", 1},
		{Int8, "int8", 1},
		{Float16, "float16", 2},
		{BFloat16, "bfloat16", 2},
		{Int16, "int16", 2},
		{Float32, "float32", 4},
		{Int32, "int32", 4},
		{Uint32, "uint32", 4},
		{Float64, "float64", 8},
		{Int64, "int64", 8},
		{Complex64, "complex64", 8},
	}
	for _, c := range cases {
		if c.d.String() != c.name {
			t.Errorf("Dtype(%d).String() = %q, want %q", c.d, c.d.String(), c.name)
		}
		if c.d.Size() != c.size {
			t.Errorf("%s.Size() = %d, want %d", c.name, c.d.Size(), c.size)
		}
		if !c.d.Valid() {
			t.Errorf("%s should be Valid", c.name)
		}
	}
	if Dtype(99).Valid() {
		t.Error("Dtype(99) should be invalid")
	}
	if Dtype(99).String() != "invalid" {
		t.Errorf("Dtype(99).String() = %q, want invalid", Dtype(99).String())
	}
}

func TestArrayConstructionAndMetadata(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	a, err := NewFloat32(data, 2, 3)
	if err != nil {
		t.Fatalf("NewFloat32: %v", err)
	}
	if a.Dtype() != Float32 {
		t.Errorf("dtype = %v, want Float32", a.Dtype())
	}
	if !reflect.DeepEqual(a.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v, want [2 3]", a.Shape())
	}
	if a.Ndim() != 2 || a.Size() != 6 {
		t.Errorf("ndim/size = %d/%d, want 2/6", a.Ndim(), a.Size())
	}
	got, err := a.ToFloat32()
	if err != nil {
		t.Fatalf("ToFloat32: %v", err)
	}
	if !reflect.DeepEqual(got, data) {
		t.Errorf("round-trip = %v, want %v", got, data)
	}
	// The returned slice must be a copy: mutating it must not touch the array.
	got[0] = 99
	again, _ := a.ToFloat32()
	if again[0] != 1 {
		t.Error("ToFloat32 did not return a copy")
	}
}

func TestArrayShapeMismatch(t *testing.T) {
	if _, err := NewFloat32([]float32{1, 2, 3}, 2, 2); err == nil {
		t.Error("expected shape-mismatch error")
	}
	if _, err := NewInt32([]int32{1, 2}, 3); err == nil {
		t.Error("expected shape-mismatch error")
	}
}

func TestZerosAndInt32(t *testing.T) {
	z, err := Zeros(Float32, 4)
	if err != nil {
		t.Fatalf("Zeros: %v", err)
	}
	vals, err := z.ToFloat32()
	if err != nil {
		t.Fatalf("ToFloat32: %v", err)
	}
	for i, v := range vals {
		if v != 0 {
			t.Errorf("zeros[%d] = %v, want 0", i, v)
		}
	}
	ia, err := NewInt32([]int32{7, 8, 9}, 3)
	if err != nil {
		t.Fatalf("NewInt32: %v", err)
	}
	iv, err := ia.ToInt32()
	if err != nil || !reflect.DeepEqual(iv, []int32{7, 8, 9}) {
		t.Errorf("int32 round-trip = %v, %v", iv, err)
	}
}

func TestFreeInvalidatesArray(t *testing.T) {
	a, _ := NewFloat32([]float32{1}, 1)
	a.Free()
	if err := a.Eval(); err == nil {
		t.Error("Eval after Free should error")
	}
	if _, err := a.ToFloat32(); err == nil {
		t.Error("ToFloat32 after Free should error")
	}
}

func TestComputeOpsUnavailableInStub(t *testing.T) {
	a, _ := NewFloat32([]float32{1, 2}, 2)
	b, _ := NewFloat32([]float32{3, 4}, 2)
	s := DefaultStream()
	ops := []func() (*Array, error){
		func() (*Array, error) { return MatMul(a, b, s) },
		func() (*Array, error) { return Add(a, b, s) },
		func() (*Array, error) { return Mul(a, b, s) },
		func() (*Array, error) { return Sub(a, b, s) },
		func() (*Array, error) { return Div(a, b, s) },
		func() (*Array, error) { return Softmax(a, -1, s) },
		func() (*Array, error) { return RMSNorm(a, b, 1e-5, s) },
		func() (*Array, error) { return Reshape(a, []int{2, 1}, s) },
		func() (*Array, error) { return Transpose(a, nil, s) },
		func() (*Array, error) { return Concatenate([]*Array{a, b}, 0, s) },
		func() (*Array, error) { return Take(a, b, 0, s) },
		func() (*Array, error) { return Argmax(a, -1, s) },
		func() (*Array, error) { return RoPE(a, 2, false, 10000, 1, 0, s) },
		func() (*Array, error) { return ScaledDotProductAttention(a, a, a, 1, nil, s) },
		func() (*Array, error) { return QuantizedMatMul(a, b, a, b, true, 64, 4, s) },
	}
	for i, op := range ops {
		if _, err := op(); !errors.Is(err, ErrMLXUnavailable) {
			t.Errorf("op %d: err = %v, want ErrMLXUnavailable", i, err)
		}
	}
}

func TestStreamAndMemoryControlsAreInert(t *testing.T) {
	s, err := NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	if err := s.Synchronize(); err != nil {
		t.Errorf("Synchronize: %v", err)
	}
	tl, _ := NewThreadLocalStream()
	SetDefaultStream(tl)
	ClearCache()
	SetWiredLimit(1 << 30)
	SetMemoryLimit(1 << 30)
	SetCacheLimit(1 << 20)
	if GetActiveMemory() != 0 || GetPeakMemory() != 0 {
		t.Error("stub memory counters should be zero")
	}
}

func BenchmarkArrayRoundTrip(b *testing.B) {
	b.ReportAllocs()
	data := make([]float32, 1024)
	for i := range data {
		data[i] = float32(i)
	}
	for b.Loop() {
		a, _ := NewFloat32(data, 1024)
		_, _ = a.ToFloat32()
		a.Free()
	}
}
