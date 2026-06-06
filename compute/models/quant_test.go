// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"errors"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

func qArr(t *testing.T, shape ...int) *mlxgo.Array {
	t.Helper()
	a, err := mlxgo.NewFloat32([]float32{1}, shape...)
	if err != nil {
		t.Fatalf("NewFloat32: %v", err)
	}
	return a
}

func qIdx(t *testing.T) *mlxgo.Array {
	t.Helper()
	a, err := mlxgo.NewInt32([]int32{0}, 1)
	if err != nil {
		t.Fatalf("NewInt32: %v", err)
	}
	return a
}

func TestParseQuantConfig(t *testing.T) {
	cases := []struct {
		name string
		cfg  string
		want quantConfig
	}{
		{"absent", `{"hidden_size":8}`, quantConfig{}},
		{"affine", `{"quantization":{"group_size":64,"bits":4,"mode":"affine"}}`, quantConfig{GroupSize: 64, Bits: 4}},
		{"no_mode", `{"quantization":{"group_size":32,"bits":8}}`, quantConfig{GroupSize: 32, Bits: 8}},
		{"mxfp4", `{"quantization":{"group_size":32,"bits":4,"mode":"mxfp4"}}`, quantConfig{}},
		{"zero_bits", `{"quantization":{"group_size":64,"bits":0}}`, quantConfig{}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := parseQuantConfig([]byte(c.cfg))
			if err != nil {
				t.Fatalf("parseQuantConfig: %v", err)
			}
			if got != c.want {
				t.Fatalf("got %+v, want %+v", got, c.want)
			}
		})
	}
}

func TestParseQuantConfigBadJSON(t *testing.T) {
	if _, err := parseQuantConfig([]byte("{")); err == nil {
		t.Fatal("expected error on malformed config")
	}
}

func TestLoadQLinearDense(t *testing.T) {
	a := qArr(t, 1)
	weights := map[string]*mlxgo.Array{"layer.proj.weight": a}
	w, err := loadQLinear(weights, "layer.proj", quantConfig{})
	if err != nil {
		t.Fatalf("loadQLinear: %v", err)
	}
	if w.isQuantized() {
		t.Fatal("dense weight reported quantized")
	}
	if w.w != a {
		t.Fatal("weight not wired through")
	}
}

func TestLoadQLinearQuantized(t *testing.T) {
	a := qArr(t, 1)
	weights := map[string]*mlxgo.Array{
		"layer.proj.weight": a,
		"layer.proj.scales": a,
		"layer.proj.biases": a,
	}
	w, err := loadQLinear(weights, "layer.proj", quantConfig{GroupSize: 64, Bits: 4})
	if err != nil {
		t.Fatalf("loadQLinear: %v", err)
	}
	if !w.isQuantized() {
		t.Fatal("quantized weight reported dense")
	}
	if w.groupSize != 64 || w.bits != 4 {
		t.Fatalf("geometry not carried: gs=%d bits=%d", w.groupSize, w.bits)
	}
}

func TestLoadQLinearScalesButDenseConfig(t *testing.T) {
	// Scales present in the checkpoint but the config selects no quantization:
	// the weight loads dense, matching a config that left this module unpacked.
	a := qArr(t, 1)
	weights := map[string]*mlxgo.Array{
		"layer.proj.weight": a,
		"layer.proj.scales": a,
		"layer.proj.biases": a,
	}
	w, err := loadQLinear(weights, "layer.proj", quantConfig{})
	if err != nil {
		t.Fatalf("loadQLinear: %v", err)
	}
	if w.isQuantized() {
		t.Fatal("loaded quantized despite dense config")
	}
}

func TestLoadQLinearMissingWeight(t *testing.T) {
	if _, err := loadQLinear(map[string]*mlxgo.Array{}, "layer.proj", quantConfig{}); err == nil {
		t.Fatal("expected error on missing weight")
	}
}

func TestLoadQLinearQuantizedMissingBiases(t *testing.T) {
	a := qArr(t, 1)
	weights := map[string]*mlxgo.Array{
		"layer.proj.weight": a,
		"layer.proj.scales": a,
	}
	if _, err := loadQLinear(weights, "layer.proj", quantConfig{GroupSize: 64, Bits: 4}); err == nil {
		t.Fatal("expected error on quantized weight missing biases")
	}
}

func TestQLinearReachesSeam(t *testing.T) {
	a := qArr(t, 1, 1)
	dense := &qLinear{w: a}
	quant := &qLinear{w: a, scales: a, biases: a, groupSize: 64, bits: 4}
	for _, c := range []struct {
		name string
		w    *qLinear
	}{{"dense", dense}, {"quantized", quant}} {
		t.Run(c.name, func(t *testing.T) {
			b := &fb{s: mlxgo.DefaultStream()}
			if out := b.qlinear(a, c.w); out != nil {
				t.Fatal("expected nil result at the stub seam")
			}
			if !errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
				t.Fatalf("expected ErrMLXUnavailable, got %v", b.err)
			}
		})
	}
}

func TestQLinearStickyError(t *testing.T) {
	a := qArr(t, 1, 1)
	b := &fb{s: mlxgo.DefaultStream(), err: errors.New("prior")}
	if out := b.qlinear(a, &qLinear{w: a}); out != nil {
		t.Fatal("expected short-circuit on sticky error")
	}
}

func TestQEmbedReachesSeam(t *testing.T) {
	a := qArr(t, 1, 1)
	idx := qIdx(t)
	for _, c := range []struct {
		name  string
		table *qLinear
	}{
		{"dense", &qLinear{w: a}},
		{"quantized", &qLinear{w: a, scales: a, biases: a, groupSize: 64, bits: 4}},
	} {
		t.Run(c.name, func(t *testing.T) {
			b := &fb{s: mlxgo.DefaultStream()}
			if out := b.qembed(c.table, idx); out != nil {
				t.Fatal("expected nil result at the stub seam")
			}
			if !errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
				t.Fatalf("expected ErrMLXUnavailable, got %v", b.err)
			}
		})
	}
}
