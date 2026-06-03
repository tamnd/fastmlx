// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"fmt"

	"github.com/tamnd/fastmlx/mlxgo"
)

// qwen3Layer holds one decoder block's weight tensors.
type qwen3Layer struct {
	inputLayernorm         *mlxgo.Array
	postAttentionLayernorm *mlxgo.Array
	qProj, kProj, vProj    *mlxgo.Array
	oProj                  *mlxgo.Array
	qNorm, kNorm           *mlxgo.Array
	gateProj, upProj       *mlxgo.Array
	downProj               *mlxgo.Array
}

// Qwen3Model is an assembled Qwen3 dense model: the decoded args plus the
// weight tensors wired into typed fields. The weights arrive as the loader's
// name-to-array map (the keys are exactly Qwen3Args.WeightNames); the
// constructor pulls each tensor into place and resolves the tied head.
type Qwen3Model struct {
	args        *Qwen3Args
	embedTokens *mlxgo.Array
	layers      []qwen3Layer
	norm        *mlxgo.Array
	lmHead      *mlxgo.Array // nil when the head is tied to the embedding table
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
	m := &Qwen3Model{args: args, layers: make([]qwen3Layer, args.NumHiddenLayers)}
	var err error
	if m.embedTokens, err = get("model.embed_tokens.weight"); err != nil {
		return nil, err
	}
	if m.norm, err = get("model.norm.weight"); err != nil {
		return nil, err
	}
	for i := range m.layers {
		p := fmt.Sprintf("model.layers.%d.", i)
		fields := []struct {
			name string
			dst  **mlxgo.Array
		}{
			{p + "input_layernorm.weight", &m.layers[i].inputLayernorm},
			{p + "post_attention_layernorm.weight", &m.layers[i].postAttentionLayernorm},
			{p + "self_attn.q_proj.weight", &m.layers[i].qProj},
			{p + "self_attn.k_proj.weight", &m.layers[i].kProj},
			{p + "self_attn.v_proj.weight", &m.layers[i].vProj},
			{p + "self_attn.o_proj.weight", &m.layers[i].oProj},
			{p + "self_attn.q_norm.weight", &m.layers[i].qNorm},
			{p + "self_attn.k_norm.weight", &m.layers[i].kNorm},
			{p + "mlp.gate_proj.weight", &m.layers[i].gateProj},
			{p + "mlp.up_proj.weight", &m.layers[i].upProj},
			{p + "mlp.down_proj.weight", &m.layers[i].downProj},
		}
		for _, f := range fields {
			if *f.dst, err = get(f.name); err != nil {
				return nil, err
			}
		}
	}
	if args.TieWordEmbeddings {
		m.lmHead = nil
	} else if m.lmHead, err = get("lm_head.weight"); err != nil {
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
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.RoPE(x, dims, false, base, 1, offset, b.s)
	b.err = err
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
	if len(caches) != len(m.layers) {
		return nil, fmt.Errorf("qwen3: got %d caches, want %d", len(caches), len(m.layers))
	}
	a := m.args
	L := len(tokens)
	eps := float32(a.RMSNormEps)
	theta := float32(a.RopeTheta)
	scale := float32(a.Scale())
	hd := a.HeadDim
	nh := a.NumAttentionHeads
	nkv := a.NumKeyValueHeads

	b := &fb{s: s}

	// Embedding lookup, then a leading batch axis: [L, D] -> [1, L, D].
	idx, err := mlxgo.NewInt32(tokens, L)
	if err != nil {
		return nil, err
	}
	h, err := mlxgo.Take(m.embedTokens, idx, 0, s)
	if err != nil {
		return nil, err
	}
	h = b.reshape(h, []int{1, L, a.HiddenSize})

	maskMode := ""
	if L > 1 {
		maskMode = "causal"
	}

	for i := range m.layers {
		layer := &m.layers[i]
		cache := caches[i]

		// Attention.
		x := b.rmsNorm(h, layer.inputLayernorm, eps)
		q := b.linear(x, layer.qProj)
		k := b.linear(x, layer.kProj)
		v := b.linear(x, layer.vProj)
		q = b.reshape(q, []int{1, L, nh, hd})
		q = b.rmsNorm(q, layer.qNorm, eps)
		q = b.transpose(q, []int{0, 2, 1, 3})
		k = b.reshape(k, []int{1, L, nkv, hd})
		k = b.rmsNorm(k, layer.kNorm, eps)
		k = b.transpose(k, []int{0, 2, 1, 3})
		v = b.reshape(v, []int{1, L, nkv, hd})
		v = b.transpose(v, []int{0, 2, 1, 3})
		offset := cache.Offset
		q = b.rope(q, hd, theta, offset)
		k = b.rope(k, hd, theta, offset)
		if b.err == nil {
			k, v, b.err = cache.Update(k, v, s)
		}
		attn := b.sdpa(q, k, v, scale, maskMode)
		attn = b.transpose(attn, []int{0, 2, 1, 3})
		attn = b.reshape(attn, []int{1, L, nh * hd})
		attn = b.linear(attn, layer.oProj)
		h = b.add(h, attn)

		// SwiGLU MLP.
		y := b.rmsNorm(h, layer.postAttentionLayernorm, eps)
		gate := b.silu(b.linear(y, layer.gateProj))
		up := b.linear(y, layer.upProj)
		y = b.linear(b.mul(gate, up), layer.downProj)
		h = b.add(h, y)
	}

	h = b.rmsNorm(h, m.norm, eps)
	head := m.lmHead
	if head == nil {
		head = m.embedTokens
	}
	logits := b.linear(h, head)
	if b.err != nil {
		return nil, b.err
	}
	return logits, nil
}
