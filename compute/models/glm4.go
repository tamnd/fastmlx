// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"github.com/tamnd/fastmlx/compute"
	"github.com/tamnd/fastmlx/mlxgo"
)

// Glm4Args decodes the dense GLM-4 config. The decoder is a Llama-shaped stack
// with three distinguishing traits the reference spells out: a sandwich-norm
// block (a post-attention norm on the attention output and a post-MLP norm on the
// MLP output, each applied before the residual add, in addition to the usual input
// and pre-MLP norms), a fused gate-and-up projection split at run time, and a
// partial, traditional rotary embedding (only the first head_dim*partial_rotary
// dims rotate, with interleaved pairing). Query, key, and value projections carry
// an optional bias; the output projection never does. The model is always untied.
type Glm4Args struct {
	ModelType             string
	HiddenSize            int
	NumHiddenLayers       int
	IntermediateSize      int
	NumAttentionHeads     int
	AttentionBias         bool
	HeadDim               int
	RMSNormEps            float64
	VocabSize             int
	NumKeyValueHeads      int
	PartialRotaryFactor   float64
	RopeTheta             float64
	RopeTraditional       bool
	MaxPositionEmbeddings int
}

type glm4Config struct {
	ModelType             string  `json:"model_type"`
	HiddenSize            int     `json:"hidden_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	IntermediateSize      int     `json:"intermediate_size"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	AttentionBias         bool    `json:"attention_bias"`
	HeadDim               *int    `json:"head_dim"`
	RMSNormEps            float64 `json:"rms_norm_eps"`
	VocabSize             int     `json:"vocab_size"`
	NumKeyValueHeads      int     `json:"num_key_value_heads"`
	PartialRotaryFactor   float64 `json:"partial_rotary_factor"`
	RopeTheta             float64 `json:"rope_theta"`
	RopeTraditional       *bool   `json:"rope_traditional"`
	MaxPositionEmbeddings *int    `json:"max_position_embeddings"`
}

// ParseGlm4Args decodes a config.json body into Glm4Args, applying the dataclass
// defaults: head_dim falls back to hidden_size/num_attention_heads, rope_traditional
// defaults true, and max_position_embeddings defaults to 32768.
func ParseGlm4Args(configJSON []byte) (*Glm4Args, error) {
	var c glm4Config
	if err := json.Unmarshal(configJSON, &c); err != nil {
		return nil, fmt.Errorf("glm4: decode config: %w", err)
	}
	if c.NumAttentionHeads <= 0 {
		return nil, fmt.Errorf("glm4: num_attention_heads must be positive, got %d", c.NumAttentionHeads)
	}
	a := &Glm4Args{
		ModelType:           c.ModelType,
		HiddenSize:          c.HiddenSize,
		NumHiddenLayers:     c.NumHiddenLayers,
		IntermediateSize:    c.IntermediateSize,
		NumAttentionHeads:   c.NumAttentionHeads,
		AttentionBias:       c.AttentionBias,
		RMSNormEps:          c.RMSNormEps,
		VocabSize:           c.VocabSize,
		NumKeyValueHeads:    c.NumKeyValueHeads,
		PartialRotaryFactor: c.PartialRotaryFactor,
		RopeTheta:           c.RopeTheta,
	}
	if c.HeadDim != nil && *c.HeadDim > 0 {
		a.HeadDim = *c.HeadDim
	} else {
		a.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	a.RopeTraditional = true
	if c.RopeTraditional != nil {
		a.RopeTraditional = *c.RopeTraditional
	}
	a.MaxPositionEmbeddings = 32768
	if c.MaxPositionEmbeddings != nil {
		a.MaxPositionEmbeddings = *c.MaxPositionEmbeddings
	}
	if err := a.validate(); err != nil {
		return nil, err
	}
	return a, nil
}

func (a *Glm4Args) validate() error {
	switch {
	case a.HiddenSize <= 0:
		return fmt.Errorf("glm4: hidden_size must be positive, got %d", a.HiddenSize)
	case a.HeadDim <= 0:
		return fmt.Errorf("glm4: head_dim must be positive, got %d", a.HeadDim)
	case a.VocabSize <= 0:
		return fmt.Errorf("glm4: vocab_size must be positive, got %d", a.VocabSize)
	case a.NumHiddenLayers <= 0:
		return fmt.Errorf("glm4: num_hidden_layers must be positive, got %d", a.NumHiddenLayers)
	case a.NumKeyValueHeads <= 0:
		return fmt.Errorf("glm4: num_key_value_heads must be positive, got %d", a.NumKeyValueHeads)
	case a.NumAttentionHeads%a.NumKeyValueHeads != 0:
		return fmt.Errorf("glm4: num_attention_heads (%d) must be a multiple of num_key_value_heads (%d)",
			a.NumAttentionHeads, a.NumKeyValueHeads)
	case a.PartialRotaryFactor <= 0 || a.PartialRotaryFactor > 1:
		return fmt.Errorf("glm4: partial_rotary_factor must be in (0,1], got %g", a.PartialRotaryFactor)
	}
	return nil
}

// NumLayers is the decoder depth.
func (a *Glm4Args) NumLayers() int { return a.NumHiddenLayers }

// Scale is the attention logit scale, head_dim raised to the -1/2 power.
func (a *Glm4Args) Scale() float64 { return math.Pow(float64(a.HeadDim), -0.5) }

// QProjOut is the query projection output width.
func (a *Glm4Args) QProjOut() int { return a.NumAttentionHeads * a.HeadDim }

// KVProjOut is the key (and value) projection output width.
func (a *Glm4Args) KVProjOut() int { return a.NumKeyValueHeads * a.HeadDim }

// GQARepeat is the grouped-query repeat factor.
func (a *Glm4Args) GQARepeat() int { return a.NumAttentionHeads / a.NumKeyValueHeads }

// RopeDims is the number of head dimensions the rotary embedding rotates:
// floor(head_dim * partial_rotary_factor), matching the reference int() truncation.
func (a *Glm4Args) RopeDims() int { return int(float64(a.HeadDim) * a.PartialRotaryFactor) }

// MakeCache builds one plain growing cache per layer.
func (a *Glm4Args) MakeCache() []*compute.KVCache {
	caches := make([]*compute.KVCache, a.NumLayers())
	for i := range caches {
		caches[i] = &compute.KVCache{}
	}
	return caches
}

// WeightNames returns the sorted parameter key set: the three attention
// projections (with bias when attention_bias is set) and the output projection,
// the fused gate-up and the down projections, the four block layernorms (input,
// post-attention, post-self-attention, post-MLP), then the embedding, the final
// norm, and the always-present lm_head.
func (a *Glm4Args) WeightNames() []string {
	names := []string{
		"model.embed_tokens.weight",
		"model.norm.weight",
		"lm_head.weight",
	}
	for i := range a.NumLayers() {
		p := fmt.Sprintf("model.layers.%d.", i)
		names = append(names,
			p+"input_layernorm.weight",
			p+"post_attention_layernorm.weight",
			p+"post_self_attn_layernorm.weight",
			p+"post_mlp_layernorm.weight",
			p+"self_attn.q_proj.weight",
			p+"self_attn.k_proj.weight",
			p+"self_attn.v_proj.weight",
			p+"self_attn.o_proj.weight",
			p+"mlp.gate_up_proj.weight",
			p+"mlp.down_proj.weight",
		)
		if a.AttentionBias {
			names = append(names,
				p+"self_attn.q_proj.bias",
				p+"self_attn.k_proj.bias",
				p+"self_attn.v_proj.bias",
			)
		}
	}
	sort.Strings(names)
	return names
}

// Sanitize is the identity: the reference GLM-4 model defines no weight rewrite,
// so the checkpoint reaches the model unchanged.
func (a *Glm4Args) Sanitize(weights map[string]*mlxgo.Array) map[string]*mlxgo.Array {
	return weights
}

// glm4Layer holds one decoder block's weights.
type glm4Layer struct {
	inputLayernorm         *mlxgo.Array
	postAttentionLayernorm *mlxgo.Array
	postSelfAttnLayernorm  *mlxgo.Array
	postMLPLayernorm       *mlxgo.Array
	qProj, kProj, vProj    *mlxgo.Array
	qBias, kBias, vBias    *mlxgo.Array // nil unless attention_bias
	oProj                  *mlxgo.Array
	gateUpProj             *mlxgo.Array
	downProj               *mlxgo.Array
}

// Glm4Model is an assembled dense GLM-4 model.
type Glm4Model struct {
	args        *Glm4Args
	embedTokens *mlxgo.Array
	layers      []glm4Layer
	norm        *mlxgo.Array
	lmHead      *mlxgo.Array
}

// NewGlm4Model wires a sanitized weight map into a runnable model.
func NewGlm4Model(args *Glm4Args, weights map[string]*mlxgo.Array) (*Glm4Model, error) {
	get := func(name string) (*mlxgo.Array, error) {
		w, ok := weights[name]
		if !ok || w == nil {
			return nil, fmt.Errorf("glm4: missing weight %q", name)
		}
		return w, nil
	}
	m := &Glm4Model{args: args, layers: make([]glm4Layer, args.NumLayers())}
	var err error
	if m.embedTokens, err = get("model.embed_tokens.weight"); err != nil {
		return nil, err
	}
	if m.norm, err = get("model.norm.weight"); err != nil {
		return nil, err
	}
	if m.lmHead, err = get("lm_head.weight"); err != nil {
		return nil, err
	}
	for i := range m.layers {
		p := fmt.Sprintf("model.layers.%d.", i)
		req := []struct {
			name string
			dst  **mlxgo.Array
		}{
			{p + "input_layernorm.weight", &m.layers[i].inputLayernorm},
			{p + "post_attention_layernorm.weight", &m.layers[i].postAttentionLayernorm},
			{p + "post_self_attn_layernorm.weight", &m.layers[i].postSelfAttnLayernorm},
			{p + "post_mlp_layernorm.weight", &m.layers[i].postMLPLayernorm},
			{p + "self_attn.q_proj.weight", &m.layers[i].qProj},
			{p + "self_attn.k_proj.weight", &m.layers[i].kProj},
			{p + "self_attn.v_proj.weight", &m.layers[i].vProj},
			{p + "self_attn.o_proj.weight", &m.layers[i].oProj},
			{p + "mlp.gate_up_proj.weight", &m.layers[i].gateUpProj},
			{p + "mlp.down_proj.weight", &m.layers[i].downProj},
		}
		for _, f := range req {
			if *f.dst, err = get(f.name); err != nil {
				return nil, err
			}
		}
		if args.AttentionBias {
			bias := []struct {
				name string
				dst  **mlxgo.Array
			}{
				{p + "self_attn.q_proj.bias", &m.layers[i].qBias},
				{p + "self_attn.k_proj.bias", &m.layers[i].kBias},
				{p + "self_attn.v_proj.bias", &m.layers[i].vBias},
			}
			for _, f := range bias {
				if *f.dst, err = get(f.name); err != nil {
					return nil, err
				}
			}
		}
	}
	return m, nil
}

// Forward runs one sequence of tokens through the model and returns the logits,
// shaped [1, len(tokens), vocab_size]. It mirrors the GLM-4 block: a sandwich-norm
// attention residual, then a sandwich-norm MLP residual whose gate-up projection
// is split at run time and combined through SwiGLU. The rotary embedding is
// partial and traditional.
func (m *Glm4Model) Forward(tokens []int32, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	return m.forwardBL(tokens, 1, len(tokens), caches, s)
}

// BatchDecode runs one decode step for batch sequences at once and returns the
// logits, shaped [batch, 1, vocab_size]. tokens holds the batch's single tokens
// in row order, the [batch, 1] decode input the reference forms with
// inputs[:, None]. Every sequence shares the same cache length (a synchronized
// batch), so with L == 1 the step needs no mask and one kernel launch serves the
// whole batch.
func (m *Glm4Model) BatchDecode(tokens []int32, batch int, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	return m.forwardBL(tokens, batch, 1, caches, s)
}

// forwardBL is the batch-polymorphic forward shared by Forward and BatchDecode.
// tokens is the row-major [batch, L] token matrix flattened to batch*L values
// and the result is [batch, L, vocab_size]; batch == 1 reproduces the
// single-sequence shapes and L == 1 is the batched decode step.
func (m *Glm4Model) forwardBL(tokens []int32, batch, L int, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	if len(caches) != len(m.layers) {
		return nil, fmt.Errorf("glm4: got %d caches, want %d", len(caches), len(m.layers))
	}
	a := m.args
	eps := float32(a.RMSNormEps)
	theta := float32(a.RopeTheta)
	scale := float32(a.Scale())
	hd := a.HeadDim
	nh := a.NumAttentionHeads
	nkv := a.NumKeyValueHeads
	ropeDims := a.RopeDims()
	trad := a.RopeTraditional

	b := &fb{s: s}

	idx, err := mlxgo.NewInt32(tokens, batch*L)
	if err != nil {
		return nil, err
	}
	h, err := mlxgo.Take(m.embedTokens, idx, 0, s)
	if err != nil {
		return nil, err
	}
	h = b.reshape(h, []int{batch, L, a.HiddenSize})

	maskMode := ""
	if L > 1 {
		maskMode = "causal"
	}

	for i := range m.layers {
		layer := &m.layers[i]
		cache := caches[i]

		x := b.rmsNorm(h, layer.inputLayernorm, eps)
		q := b.linearBias(x, layer.qProj, layer.qBias)
		k := b.linearBias(x, layer.kProj, layer.kBias)
		v := b.linearBias(x, layer.vProj, layer.vBias)
		q = b.transpose(b.reshape(q, []int{batch, L, nh, hd}), []int{0, 2, 1, 3})
		k = b.transpose(b.reshape(k, []int{batch, L, nkv, hd}), []int{0, 2, 1, 3})
		v = b.transpose(b.reshape(v, []int{batch, L, nkv, hd}), []int{0, 2, 1, 3})
		offset := cache.Offset
		q = b.ropeTrad(q, ropeDims, trad, theta, offset)
		k = b.ropeTrad(k, ropeDims, trad, theta, offset)
		if b.err == nil {
			k, v, b.err = cache.Update(k, v, s)
		}
		attn := b.sdpa(q, k, v, scale, maskMode)
		attn = b.reshape(b.transpose(attn, []int{0, 2, 1, 3}), []int{batch, L, nh * hd})
		attn = b.linear(attn, layer.oProj)
		// Sandwich norm: normalize the attention output before the residual add.
		h = b.add(h, b.rmsNorm(attn, layer.postSelfAttnLayernorm, eps))

		y := b.rmsNorm(h, layer.postAttentionLayernorm, eps)
		mlp := b.glmMLP(y, layer)
		// Sandwich norm: normalize the MLP output before the residual add.
		h = b.add(h, b.rmsNorm(mlp, layer.postMLPLayernorm, eps))
	}

	h = b.rmsNorm(h, m.norm, eps)
	logits := b.linear(h, m.lmHead)
	if b.err != nil {
		return nil, b.err
	}
	return logits, nil
}

// glmMLP runs the fused gate-up projection, splits it into the gate and up halves,
// and combines them through SwiGLU before the down projection.
func (b *fb) glmMLP(x *mlxgo.Array, layer *glm4Layer) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	gu := b.linear(x, layer.gateUpProj)
	if b.err != nil {
		return nil
	}
	parts, err := mlxgo.Split(gu, 2, -1, b.s)
	if err != nil {
		b.err = err
		return nil
	}
	gate, up := parts[0], parts[1]
	return b.linear(b.mul(b.silu(gate), up), layer.downProj)
}

// LoadGlm4 assembles a runnable model from a checkpoint.
func LoadGlm4(configJSON, blob []byte) (*Glm4Model, error) {
	args, err := ParseGlm4Args(configJSON)
	if err != nil {
		return nil, err
	}
	weights, err := compute.LoadTensors(blob)
	if err != nil {
		return nil, err
	}
	return NewGlm4Model(args, args.Sanitize(weights))
}
