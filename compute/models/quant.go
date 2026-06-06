// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/json"
	"fmt"

	"github.com/tamnd/fastmlx/mlxgo"
)

// quantConfig is the affine quantization geometry a checkpoint was packed with,
// read from the top-level "quantization" config block. A zero value (Bits 0) marks
// an unquantized checkpoint, where every weight loads dense.
type quantConfig struct {
	GroupSize int
	Bits      int
}

// quantized reports whether the config selects affine quantization.
func (q quantConfig) quantized() bool { return q.Bits > 0 }

// parseQuantConfig reads the top-level affine quantization geometry from a model
// config. An absent "quantization" block yields a zero quantConfig, so the model
// loads dense. A non-affine mode (mxfp4) also yields a zero value: the pinned
// backend runs only the affine path, so such a checkpoint is left for a later
// mlx-c bump rather than decoded with the wrong kernel.
func parseQuantConfig(configJSON []byte) (quantConfig, error) {
	var c struct {
		Quantization *struct {
			GroupSize int    `json:"group_size"`
			Bits      int    `json:"bits"`
			Mode      string `json:"mode"`
		} `json:"quantization"`
	}
	if err := json.Unmarshal(configJSON, &c); err != nil {
		return quantConfig{}, err
	}
	if c.Quantization == nil || c.Quantization.Bits == 0 {
		return quantConfig{}, nil
	}
	if c.Quantization.Mode != "" && c.Quantization.Mode != "affine" {
		return quantConfig{}, nil
	}
	return quantConfig{GroupSize: c.Quantization.GroupSize, Bits: c.Quantization.Bits}, nil
}

// qLinear is a linear weight that may be affine-quantized. A dense weight sets only
// w (the [out, in] matrix); a quantized weight carries the packed weight plus its
// per-group scales and biases and the group size and bit width nn.QuantizedLinear
// was built with, the triple the QuantizedMatMul kernel reads. The presence of the
// sibling scales in the checkpoint is the per-module quantization signal, matching
// the reference loader's class_predicate.
type qLinear struct {
	w, scales, biases *mlxgo.Array
	groupSize, bits   int
}

// isQuantized reports whether this weight carries the affine triple.
func (q *qLinear) isQuantized() bool { return q != nil && q.scales != nil }

// loadQLinear resolves a linear weight by its module name (without the trailing
// ".weight"), returning it dense or, when the config selects quantization and the
// checkpoint carries the sibling scales for that module, quantized at the config
// group size and bit width. A quantized module must carry its biases too. The scales
// key is the per-module signal, the same test the reference loader applies, so a
// checkpoint that quantizes only some modules loads each at its own kind.
func loadQLinear(weights map[string]*mlxgo.Array, name string, q quantConfig) (*qLinear, error) {
	w, ok := weights[name+".weight"]
	if !ok || w == nil {
		return nil, fmt.Errorf("models: missing weight %q", name+".weight")
	}
	lw := &qLinear{w: w}
	if scales, ok := weights[name+".scales"]; q.quantized() && ok {
		biases, ok := weights[name+".biases"]
		if !ok || biases == nil {
			return nil, fmt.Errorf("models: quantized %q missing biases", name)
		}
		lw.scales = scales
		lw.biases = biases
		lw.groupSize = q.GroupSize
		lw.bits = q.Bits
	}
	return lw, nil
}

// qlinear computes x @ w.T for a possibly-quantized weight. A dense weight takes the
// plain matmul (transposing the [out, in] weight the way nn.Linear does); a quantized
// weight takes QuantizedMatMul with transpose set, which consumes the packed weight in
// its native layout with no host transpose.
func (b *fb) qlinear(x *mlxgo.Array, w *qLinear) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	if !w.isQuantized() {
		return b.linear(x, w.w)
	}
	r, err := mlxgo.QuantizedMatMul(x, w.w, w.scales, w.biases, true, w.groupSize, w.bits, b.s)
	b.err = err
	return r
}

// qlinearBias adds a bias to the qlinear result, broadcasting over the leading axes
// the same as nn.Linear with bias. A nil bias returns the bare projection.
func (b *fb) qlinearBias(x *mlxgo.Array, w *qLinear, bias *mlxgo.Array) *mlxgo.Array {
	out := b.qlinear(x, w)
	if bias == nil {
		return out
	}
	return b.add(out, bias)
}

// qembed gathers token rows from a possibly-quantized embedding table. A dense table
// is a plain row take; a quantized table (nn.QuantizedEmbedding) gathers the packed
// rows and their scales and biases and dequantizes the gathered block, matching the
// reference QuantizedEmbedding.__call__.
func (b *fb) qembed(table *qLinear, idx *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	rows, err := mlxgo.Take(table.w, idx, 0, b.s)
	if err != nil {
		b.err = err
		return nil
	}
	if !table.isQuantized() {
		return rows
	}
	sc, err := mlxgo.Take(table.scales, idx, 0, b.s)
	if err != nil {
		b.err = err
		return nil
	}
	bi, err := mlxgo.Take(table.biases, idx, 0, b.s)
	if err != nil {
		b.err = err
		return nil
	}
	r, err := mlxgo.Dequantize(rows, sc, bi, table.groupSize, table.bits, b.s)
	b.err = err
	return r
}
