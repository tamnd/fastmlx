// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"fmt"

	"github.com/tamnd/fastmlx/mlxgo"
)

// qwen3Layer holds one decoder block's weight tensors. The projections and the
// MLP weights may be affine-quantized (loaded as qLinear, dense when the
// checkpoint ships no scales for them); the per-head RMSNorm weights are never
// quantized and stay plain arrays.
type qwen3Layer struct {
	inputLayernorm         *mlxgo.Array
	postAttentionLayernorm *mlxgo.Array
	qProj, kProj, vProj    *qLinear
	oProj                  *qLinear
	qNorm, kNorm           *mlxgo.Array
	gateProj, upProj       *qLinear
	downProj               *qLinear
}

// Qwen3Model is an assembled Qwen3 dense model: the decoded args plus the
// weight tensors wired into typed fields. The weights arrive as the loader's
// name-to-array map (the keys are exactly Qwen3Args.WeightNames); the
// constructor pulls each tensor into place and resolves the tied head.
type Qwen3Model struct {
	args        *Qwen3Args
	embedTokens *qLinear
	layers      []qwen3Layer
	norm        *mlxgo.Array
	lmHead      *qLinear // nil when the head is tied to the embedding table
}

// NewQwen3Model wires a sanitized weight map into a runnable model. Every key in
// Qwen3Args.WeightNames must be present; a tied model has no lm_head.weight and
// reuses the embedding table for the output projection.
func NewQwen3Model(args *Qwen3Args, weights map[string]*mlxgo.Array) (*Qwen3Model, error) {
	get := func(name string) (*mlxgo.Array, error) {
		w, ok := weights[name]
		if !ok || w == nil {
			return nil, fmt.Errorf("qwen3: missing weight %q", name)
		}
		return w, nil
	}
	// getQ resolves a possibly-quantized weight by its module name (no ".weight"
	// suffix); a module that ships scales loads quantized at the config geometry.
	getQ := func(name string) (*qLinear, error) {
		return loadQLinear(weights, name, args.quant)
	}
	m := &Qwen3Model{args: args, layers: make([]qwen3Layer, args.NumHiddenLayers)}
	var err error
	if m.embedTokens, err = getQ("model.embed_tokens"); err != nil {
		return nil, err
	}
	if m.norm, err = get("model.norm.weight"); err != nil {
		return nil, err
	}
	for i := range m.layers {
		p := fmt.Sprintf("model.layers.%d.", i)
		norms := []struct {
			name string
			dst  **mlxgo.Array
		}{
			{p + "input_layernorm.weight", &m.layers[i].inputLayernorm},
			{p + "post_attention_layernorm.weight", &m.layers[i].postAttentionLayernorm},
			{p + "self_attn.q_norm.weight", &m.layers[i].qNorm},
			{p + "self_attn.k_norm.weight", &m.layers[i].kNorm},
		}
		for _, f := range norms {
			if *f.dst, err = get(f.name); err != nil {
				return nil, err
			}
		}
		projs := []struct {
			name string
			dst  **qLinear
		}{
			{p + "self_attn.q_proj", &m.layers[i].qProj},
			{p + "self_attn.k_proj", &m.layers[i].kProj},
			{p + "self_attn.v_proj", &m.layers[i].vProj},
			{p + "self_attn.o_proj", &m.layers[i].oProj},
			{p + "mlp.gate_proj", &m.layers[i].gateProj},
			{p + "mlp.up_proj", &m.layers[i].upProj},
			{p + "mlp.down_proj", &m.layers[i].downProj},
		}
		for _, f := range projs {
			if *f.dst, err = getQ(f.name); err != nil {
				return nil, err
			}
		}
	}
	if args.TieWordEmbeddings {
		m.lmHead = nil
	} else if m.lmHead, err = getQ("lm_head"); err != nil {
		return nil, err
	}
	return m, nil
}

// fb chains mlxgo ops with a sticky error so the forward reads top to bottom.
// Once an op fails (always at the first kernel under the stub build), every
// later op short-circuits and the error surfaces from Forward.
type fb struct {
	s   *mlxgo.Stream
	err error
}

func (b *fb) linear(x, w *mlxgo.Array) *mlxgo.Array {
	// nn.Linear computes x @ w.T for a weight stored [out, in].
	wt := b.transpose(w, []int{1, 0})
	return b.matmul(x, wt)
}

func (b *fb) matmul(x, w *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.MatMul(x, w, b.s)
	b.err = err
	return r
}

func (b *fb) transpose(x *mlxgo.Array, axes []int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Transpose(x, axes, b.s)
	b.err = err
	return r
}

func (b *fb) reshape(x *mlxgo.Array, shape []int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Reshape(x, shape, b.s)
	b.err = err
	return r
}

func (b *fb) rmsNorm(x, w *mlxgo.Array, eps float32) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.RMSNorm(x, w, eps, b.s)
	b.err = err
	return r
}

func (b *fb) rope(x *mlxgo.Array, dims int, base float32, offset int) *mlxgo.Array {
	return b.ropeTrad(x, dims, false, base, offset)
}

// ropeTrad is rope with an explicit traditional (interleaved) flag. GLM rotates
// only the first `dims` of each head with traditional pairing.
func (b *fb) ropeTrad(x *mlxgo.Array, dims int, traditional bool, base float32, offset int) *mlxgo.Array {
	return b.ropeScaled(x, dims, traditional, base, 1, offset)
}

// ropeScaled is rope with an explicit position scale. Phi-4's linear rope
// scaling passes 1/factor so positions compress by the configured factor.
func (b *fb) ropeScaled(x *mlxgo.Array, dims int, traditional bool, base, scale float32, offset int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.RoPE(x, dims, traditional, base, scale, offset, b.s)
	b.err = err
	return r
}

// ropePerRow applies RoPE with a per-row position offset, the path a left-padded
// ragged cohort needs: every row's real tokens start at a different position
// (leftPad[b] tokens behind the padded cursor), so each row rotates at its own
// offset. The binding's mlx_fast_rope takes a scalar offset, so this splits the
// batch on axis 0, ropes each row at its offset through apply (the same scalar
// rope variant the uniform path uses, closed over its dims, base, scale, and
// traditional flag), and concatenates the rows back. The caller engages it only
// when the cache reports per-row offsets; a uniform cohort keeps the single
// scalar-offset rope launch. offsets has one entry per batch row.
func (b *fb) ropePerRow(x *mlxgo.Array, offsets []int, apply func(row *mlxgo.Array, offset int) *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	batch := len(offsets)
	if batch == 1 {
		return apply(x, offsets[0])
	}
	rows, err := mlxgo.Split(x, batch, 0, b.s)
	if err != nil {
		b.err = err
		return nil
	}
	out := make([]*mlxgo.Array, batch)
	for i, row := range rows {
		out[i] = apply(row, offsets[i])
		if b.err != nil {
			return nil
		}
	}
	r, err := mlxgo.Concatenate(out, 0, b.s)
	if err != nil {
		b.err = err
		return nil
	}
	return r
}

func (b *fb) add(x, y *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Add(x, y, b.s)
	b.err = err
	return r
}

func (b *fb) mul(x, y *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Mul(x, y, b.s)
	b.err = err
	return r
}

func (b *fb) silu(x *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	sig, err := mlxgo.Sigmoid(x, b.s)
	if err != nil {
		b.err = err
		return nil
	}
	return b.mul(x, sig)
}

func (b *fb) sdpa(q, k, v *mlxgo.Array, scale float32, maskMode string) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.ScaledDotProductAttention(q, k, v, scale, maskMode, nil, b.s)
	b.err = err
	return r
}

// Forward runs a single batch (one sequence) of tokens through the model and
// returns the logits, shaped [1, len(tokens), vocab_size]. caches must hold one
// KVTensorCache per layer (from a per-sequence allocation); each layer reads its
// pre-step offset for RoPE, then appends its keys and values.
func (m *Qwen3Model) Forward(tokens []int32, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	return m.forwardBL(tokens, 1, len(tokens), caches, s)
}

// BatchDecode runs one decode step for batch sequences at once and returns the
// logits, shaped [batch, 1, vocab_size]. tokens holds the batch's single tokens
// in row order, the [batch, 1] decode input the reference forms with
// inputs[:, None]. Every sequence shares the same cache length (a synchronized
// batch), so with L == 1 the step needs no mask and one kernel launch serves the
// whole batch.
func (m *Qwen3Model) BatchDecode(tokens []int32, batch int, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	return m.forwardBL(tokens, batch, 1, caches, s)
}

// forwardBL is the batch-polymorphic forward shared by Forward and BatchDecode.
// tokens is the row-major [batch, L] token matrix flattened to batch*L values
// and the result is [batch, L, vocab_size]; batch == 1 reproduces the
// single-sequence shapes and L == 1 is the batched decode step.
func (m *Qwen3Model) forwardBL(tokens []int32, batch, L int, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	if len(caches) != len(m.layers) {
		return nil, fmt.Errorf("qwen3: got %d caches, want %d", len(caches), len(m.layers))
	}
	a := m.args
	eps := float32(a.RMSNormEps)
	theta := float32(a.RopeTheta)
	scale := float32(a.Scale())
	hd := a.HeadDim
	nh := a.NumAttentionHeads
	nkv := a.NumKeyValueHeads

	b := &fb{s: s}

	// Embedding lookup, then a leading batch axis: [batch*L, D] -> [batch, L, D].
	idx, err := mlxgo.NewInt32(tokens, batch*L)
	if err != nil {
		return nil, err
	}
	h := b.qembed(m.embedTokens, idx)
	h = b.reshape(h, []int{batch, L, a.HiddenSize})

	mode, mask, err := caches[0].AttnMask(batch, L, s)
	if err != nil {
		return nil, err
	}
	ropeOff := caches[0].RopeOffsets()

	for i := range m.layers {
		layer := &m.layers[i]
		cache := caches[i]

		// Attention.
		x := b.rmsNorm(h, layer.inputLayernorm, eps)
		q := b.qlinear(x, layer.qProj)
		k := b.qlinear(x, layer.kProj)
		v := b.qlinear(x, layer.vProj)
		q = b.reshape(q, []int{batch, L, nh, hd})
		q = b.rmsNorm(q, layer.qNorm, eps)
		q = b.transpose(q, []int{0, 2, 1, 3})
		k = b.reshape(k, []int{batch, L, nkv, hd})
		k = b.rmsNorm(k, layer.kNorm, eps)
		k = b.transpose(k, []int{0, 2, 1, 3})
		v = b.reshape(v, []int{batch, L, nkv, hd})
		v = b.transpose(v, []int{0, 2, 1, 3})
		offset := cache.Offset
		if ropeOff == nil {
			q = b.rope(q, hd, theta, offset)
			k = b.rope(k, hd, theta, offset)
		} else {
			q = b.ropePerRow(q, ropeOff, func(r *mlxgo.Array, o int) *mlxgo.Array { return b.rope(r, hd, theta, o) })
			k = b.ropePerRow(k, ropeOff, func(r *mlxgo.Array, o int) *mlxgo.Array { return b.rope(r, hd, theta, o) })
		}
		if b.err == nil {
			k, v, b.err = cache.Update(k, v, s)
		}
		attn := b.sdpaWith(q, k, v, scale, mode, mask)
		attn = b.transpose(attn, []int{0, 2, 1, 3})
		attn = b.reshape(attn, []int{batch, L, nh * hd})
		attn = b.qlinear(attn, layer.oProj)
		h = b.add(h, attn)

		// SwiGLU MLP.
		y := b.rmsNorm(h, layer.postAttentionLayernorm, eps)
		gate := b.silu(b.qlinear(y, layer.gateProj))
		up := b.qlinear(y, layer.upProj)
		y = b.qlinear(b.mul(gate, up), layer.downProj)
		h = b.add(h, y)
	}

	h = b.rmsNorm(h, m.norm, eps)
	head := m.lmHead
	if head == nil {
		head = m.embedTokens
	}
	logits := b.qlinear(h, head)
	if b.err != nil {
		return nil, b.err
	}
	return logits, nil
}
