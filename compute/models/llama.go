// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/tamnd/fastmlx/compute"
	"github.com/tamnd/fastmlx/mlxgo"
)

// LlamaArgs decodes the config of a Llama 3.x dense model. It is the same
// decoder stack as Qwen3 with three differences: there are no per-head query
// and key RMSNorms, the projections may carry bias terms, and the RoPE has its
// own theta, traditional flag, and scaling variants. The defaults follow the
// reference dataclass: head_dim and num_key_value_heads derive from the head
// count, rope_theta is 10000 when absent, and the embeddings are tied unless
// the config says otherwise.
type LlamaArgs struct {
	ModelType             string
	HiddenSize            int
	NumHiddenLayers       int
	IntermediateSize      int
	NumAttentionHeads     int
	RMSNormEps            float64
	VocabSize             int
	NumKeyValueHeads      int
	HeadDim               int
	MaxPositionEmbeddings int
	RopeTheta             float64
	RopeTraditional       bool
	RopeScaling           RopeScaling
	AttentionBias         bool
	MLPBias               bool
	TieWordEmbeddings     bool
}

// llamaConfig is the raw JSON shape, with pointers for the fields whose absence
// means "use the dataclass default" rather than the Go zero value.
type llamaConfig struct {
	ModelType             string      `json:"model_type"`
	HiddenSize            int         `json:"hidden_size"`
	NumHiddenLayers       int         `json:"num_hidden_layers"`
	IntermediateSize      int         `json:"intermediate_size"`
	NumAttentionHeads     int         `json:"num_attention_heads"`
	RMSNormEps            float64     `json:"rms_norm_eps"`
	VocabSize             int         `json:"vocab_size"`
	NumKeyValueHeads      *int        `json:"num_key_value_heads"`
	HeadDim               *int        `json:"head_dim"`
	MaxPositionEmbeddings int         `json:"max_position_embeddings"`
	RopeTheta             *float64    `json:"rope_theta"`
	RopeTraditional       bool        `json:"rope_traditional"`
	RopeScaling           RopeScaling `json:"rope_scaling"`
	AttentionBias         bool        `json:"attention_bias"`
	MLPBias               bool        `json:"mlp_bias"`
	TieWordEmbeddings     *bool       `json:"tie_word_embeddings"`
}

// ParseLlamaArgs decodes a config.json body into LlamaArgs, applying the
// dataclass defaults and validating the dimensions the forward pass relies on.
func ParseLlamaArgs(configJSON []byte) (*LlamaArgs, error) {
	var c llamaConfig
	if err := json.Unmarshal(configJSON, &c); err != nil {
		return nil, fmt.Errorf("llama: decode config: %w", err)
	}
	if c.NumAttentionHeads <= 0 {
		return nil, fmt.Errorf("llama: num_attention_heads must be positive, got %d", c.NumAttentionHeads)
	}
	a := &LlamaArgs{
		ModelType:             c.ModelType,
		HiddenSize:            c.HiddenSize,
		NumHiddenLayers:       c.NumHiddenLayers,
		IntermediateSize:      c.IntermediateSize,
		NumAttentionHeads:     c.NumAttentionHeads,
		RMSNormEps:            c.RMSNormEps,
		VocabSize:             c.VocabSize,
		MaxPositionEmbeddings: c.MaxPositionEmbeddings,
		RopeTraditional:       c.RopeTraditional,
		RopeScaling:           c.RopeScaling,
		AttentionBias:         c.AttentionBias,
		MLPBias:               c.MLPBias,
	}
	if c.HeadDim != nil {
		a.HeadDim = *c.HeadDim
	} else {
		a.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	if c.NumKeyValueHeads != nil {
		a.NumKeyValueHeads = *c.NumKeyValueHeads
	} else {
		a.NumKeyValueHeads = c.NumAttentionHeads
	}
	if c.RopeTheta != nil {
		a.RopeTheta = *c.RopeTheta
	} else {
		a.RopeTheta = 10000
	}
	if c.TieWordEmbeddings != nil {
		a.TieWordEmbeddings = *c.TieWordEmbeddings
	} else {
		a.TieWordEmbeddings = true
	}
	if err := a.validate(); err != nil {
		return nil, err
	}
	return a, nil
}

func (a *LlamaArgs) validate() error {
	switch {
	case a.HiddenSize <= 0:
		return fmt.Errorf("llama: hidden_size must be positive, got %d", a.HiddenSize)
	case a.NumHiddenLayers <= 0:
		return fmt.Errorf("llama: num_hidden_layers must be positive, got %d", a.NumHiddenLayers)
	case a.HeadDim <= 0:
		return fmt.Errorf("llama: head_dim must be positive, got %d", a.HeadDim)
	case a.VocabSize <= 0:
		return fmt.Errorf("llama: vocab_size must be positive, got %d", a.VocabSize)
	case a.NumAttentionHeads%a.NumKeyValueHeads != 0:
		return fmt.Errorf("llama: num_attention_heads (%d) must be a multiple of num_key_value_heads (%d)",
			a.NumAttentionHeads, a.NumKeyValueHeads)
	}
	return nil
}

// Scale is the attention logit scale, head_dim raised to the -1/2 power.
func (a *LlamaArgs) Scale() float64 { return math.Pow(float64(a.HeadDim), -0.5) }

// QProjOut is the query projection output width.
func (a *LlamaArgs) QProjOut() int { return a.NumAttentionHeads * a.HeadDim }

// KVProjOut is the key (and value) projection output width.
func (a *LlamaArgs) KVProjOut() int { return a.NumKeyValueHeads * a.HeadDim }

// GQARepeat is the grouped-query repeat factor.
func (a *LlamaArgs) GQARepeat() int { return a.NumAttentionHeads / a.NumKeyValueHeads }

// MakeCache builds one plain growing KV cache per layer.
func (a *LlamaArgs) MakeCache() []*compute.KVCache {
	caches := make([]*compute.KVCache, a.NumHiddenLayers)
	for i := range caches {
		caches[i] = &compute.KVCache{}
	}
	return caches
}

// WeightNames returns the sorted parameter key set. Each layer has the four
// attention projections and the three MLP projections (with .bias keys when the
// config enables attention or MLP bias) plus the two layernorms; the model has
// the embedding, the final norm, and an untied lm_head.
func (a *LlamaArgs) WeightNames() []string {
	names := []string{"model.embed_tokens.weight"}
	for i := range a.NumHiddenLayers {
		p := fmt.Sprintf("model.layers.%d.", i)
		names = append(names,
			p+"input_layernorm.weight",
			p+"post_attention_layernorm.weight",
			p+"self_attn.q_proj.weight",
			p+"self_attn.k_proj.weight",
			p+"self_attn.v_proj.weight",
			p+"self_attn.o_proj.weight",
			p+"mlp.gate_proj.weight",
			p+"mlp.up_proj.weight",
			p+"mlp.down_proj.weight",
		)
		if a.AttentionBias {
			names = append(names,
				p+"self_attn.q_proj.bias",
				p+"self_attn.k_proj.bias",
				p+"self_attn.v_proj.bias",
				p+"self_attn.o_proj.bias",
			)
		}
		if a.MLPBias {
			names = append(names,
				p+"mlp.gate_proj.bias",
				p+"mlp.up_proj.bias",
				p+"mlp.down_proj.bias",
			)
		}
	}
	names = append(names, "model.norm.weight")
	if !a.TieWordEmbeddings {
		names = append(names, "lm_head.weight")
	}
	sort.Strings(names)
	return names
}

// Sanitize drops the weights the model must not receive: the precomputed rotary
// inverse-frequency buffers some checkpoints ship, and a tied checkpoint's
// explicit lm_head.weight.
func (a *LlamaArgs) Sanitize(weights map[string]*mlxgo.Array) map[string]*mlxgo.Array {
	for k := range weights {
		if strings.Contains(k, "self_attn.rotary_emb.inv_freq") {
			delete(weights, k)
		}
	}
	if a.TieWordEmbeddings {
		delete(weights, "lm_head.weight")
	}
	return weights
}

// llamaLayer holds one decoder block's weights, with optional projection biases.
type llamaLayer struct {
	inputLayernorm         *mlxgo.Array
	postAttentionLayernorm *mlxgo.Array
	qProj, kProj, vProj    *mlxgo.Array
	oProj                  *mlxgo.Array
	qBias, kBias, vBias    *mlxgo.Array
	oBias                  *mlxgo.Array
	gateProj, upProj       *mlxgo.Array
	downProj               *mlxgo.Array
	gateBias, upBias       *mlxgo.Array
	downBias               *mlxgo.Array
}

// LlamaModel is an assembled Llama dense model.
type LlamaModel struct {
	args        *LlamaArgs
	embedTokens *mlxgo.Array
	layers      []llamaLayer
	norm        *mlxgo.Array
	lmHead      *mlxgo.Array // nil when tied
}

// NewLlamaModel wires a sanitized weight map into a runnable model.
func NewLlamaModel(args *LlamaArgs, weights map[string]*mlxgo.Array) (*LlamaModel, error) {
	get := func(name string) (*mlxgo.Array, error) {
		w, ok := weights[name]
		if !ok || w == nil {
			return nil, fmt.Errorf("llama: missing weight %q", name)
		}
		return w, nil
	}
	m := &LlamaModel{args: args, layers: make([]llamaLayer, args.NumHiddenLayers)}
	var err error
	if m.embedTokens, err = get("model.embed_tokens.weight"); err != nil {
		return nil, err
	}
	if m.norm, err = get("model.norm.weight"); err != nil {
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
			{p + "self_attn.q_proj.weight", &m.layers[i].qProj},
			{p + "self_attn.k_proj.weight", &m.layers[i].kProj},
			{p + "self_attn.v_proj.weight", &m.layers[i].vProj},
			{p + "self_attn.o_proj.weight", &m.layers[i].oProj},
			{p + "mlp.gate_proj.weight", &m.layers[i].gateProj},
			{p + "mlp.up_proj.weight", &m.layers[i].upProj},
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
				{p + "self_attn.o_proj.bias", &m.layers[i].oBias},
			}
			for _, f := range bias {
				if *f.dst, err = get(f.name); err != nil {
					return nil, err
				}
			}
		}
		if args.MLPBias {
			bias := []struct {
				name string
				dst  **mlxgo.Array
			}{
				{p + "mlp.gate_proj.bias", &m.layers[i].gateBias},
				{p + "mlp.up_proj.bias", &m.layers[i].upBias},
				{p + "mlp.down_proj.bias", &m.layers[i].downBias},
			}
			for _, f := range bias {
				if *f.dst, err = get(f.name); err != nil {
					return nil, err
				}
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

// linearBias applies x @ w.T and adds the bias when present.
func (b *fb) linearBias(x, w, bias *mlxgo.Array) *mlxgo.Array {
	y := b.linear(x, w)
	if bias != nil {
		y = b.add(y, bias)
	}
	return y
}

// Forward runs one sequence of tokens through the model and returns the logits,
// shaped [1, len(tokens), vocab_size]. It mirrors the Qwen3 forward without the
// per-head q/k norms and with optional projection biases.
func (m *LlamaModel) Forward(tokens []int32, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	if len(caches) != len(m.layers) {
		return nil, fmt.Errorf("llama: got %d caches, want %d", len(caches), len(m.layers))
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

		x := b.rmsNorm(h, layer.inputLayernorm, eps)
		q := b.linearBias(x, layer.qProj, layer.qBias)
		k := b.linearBias(x, layer.kProj, layer.kBias)
		v := b.linearBias(x, layer.vProj, layer.vBias)
		q = b.transpose(b.reshape(q, []int{1, L, nh, hd}), []int{0, 2, 1, 3})
		k = b.transpose(b.reshape(k, []int{1, L, nkv, hd}), []int{0, 2, 1, 3})
		v = b.transpose(b.reshape(v, []int{1, L, nkv, hd}), []int{0, 2, 1, 3})
		offset := cache.Offset
		q = b.rope(q, hd, theta, offset)
		k = b.rope(k, hd, theta, offset)
		if b.err == nil {
			k, v, b.err = cache.Update(k, v, s)
		}
		attn := b.sdpa(q, k, v, scale, maskMode)
		attn = b.reshape(b.transpose(attn, []int{0, 2, 1, 3}), []int{1, L, nh * hd})
		attn = b.linearBias(attn, layer.oProj, layer.oBias)
		h = b.add(h, attn)

		y := b.rmsNorm(h, layer.postAttentionLayernorm, eps)
		gate := b.silu(b.linearBias(y, layer.gateProj, layer.gateBias))
		up := b.linearBias(y, layer.upProj, layer.upBias)
		y = b.linearBias(b.mul(gate, up), layer.downProj, layer.downBias)
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

// LoadLlama assembles a runnable model from a checkpoint: decode the config,
// load the weights, drop the tied head and rotary buffers, and wire the result.
func LoadLlama(configJSON, blob []byte) (*LlamaModel, error) {
	args, err := ParseLlamaArgs(configJSON)
	if err != nil {
		return nil, err
	}
	weights, err := compute.LoadTensors(blob)
	if err != nil {
		return nil, err
	}
	return NewLlamaModel(args, args.Sanitize(weights))
}
