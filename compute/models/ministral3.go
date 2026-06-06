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

// MinistralArgs decodes the Ministral 3 text config. It is the Llama dense
// decoder with three additions: the rotary parameters arrive in a nested
// rope_parameters object that also carries the llama4 attention-scale terms, the
// layers carry a per-layer attention kind (full or sliding window), and the
// queries are multiplied by a position-dependent scale before attention. There
// are no projection biases. The number of decoder layers is the length of
// layer_types; num_hidden_layers is informational and layer_types defaults to
// that many full-attention layers.
type MinistralArgs struct {
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
	TieWordEmbeddings     bool
	SlidingWindow         int
	LayerTypes            []string

	// Flattened rope_parameters.
	RopeTheta                     float64
	Llama4ScalingBeta             float64
	OriginalMaxPositionEmbeddings int
}

const (
	fullAttention    = "full_attention"
	slidingAttention = "sliding_attention"
)

type ministralConfig struct {
	ModelType             string         `json:"model_type"`
	HiddenSize            int            `json:"hidden_size"`
	NumHiddenLayers       int            `json:"num_hidden_layers"`
	IntermediateSize      int            `json:"intermediate_size"`
	NumAttentionHeads     int            `json:"num_attention_heads"`
	RMSNormEps            float64        `json:"rms_norm_eps"`
	VocabSize             int            `json:"vocab_size"`
	NumKeyValueHeads      *int           `json:"num_key_value_heads"`
	HeadDim               *int           `json:"head_dim"`
	MaxPositionEmbeddings int            `json:"max_position_embeddings"`
	TieWordEmbeddings     *bool          `json:"tie_word_embeddings"`
	SlidingWindow         int            `json:"sliding_window"`
	LayerTypes            []string       `json:"layer_types"`
	RopeParameters        map[string]any `json:"rope_parameters"`
}

// ParseMinistralArgs decodes a config.json body into MinistralArgs, applying the
// dataclass defaults: num_key_value_heads and head_dim derive from the head
// count, tie_word_embeddings defaults to true, and layer_types defaults to
// num_hidden_layers full-attention layers.
func ParseMinistralArgs(configJSON []byte) (*MinistralArgs, error) {
	var c ministralConfig
	if err := json.Unmarshal(configJSON, &c); err != nil {
		return nil, fmt.Errorf("ministral3: decode config: %w", err)
	}
	if c.NumAttentionHeads <= 0 {
		return nil, fmt.Errorf("ministral3: num_attention_heads must be positive, got %d", c.NumAttentionHeads)
	}
	if c.RopeParameters == nil {
		return nil, fmt.Errorf("ministral3: rope_parameters is required")
	}
	a := &MinistralArgs{
		ModelType:             c.ModelType,
		HiddenSize:            c.HiddenSize,
		NumHiddenLayers:       c.NumHiddenLayers,
		IntermediateSize:      c.IntermediateSize,
		NumAttentionHeads:     c.NumAttentionHeads,
		RMSNormEps:            c.RMSNormEps,
		VocabSize:             c.VocabSize,
		MaxPositionEmbeddings: c.MaxPositionEmbeddings,
		SlidingWindow:         c.SlidingWindow,
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
	if c.TieWordEmbeddings != nil {
		a.TieWordEmbeddings = *c.TieWordEmbeddings
	} else {
		a.TieWordEmbeddings = true
	}
	if c.LayerTypes != nil {
		a.LayerTypes = c.LayerTypes
	} else {
		a.LayerTypes = make([]string, c.NumHiddenLayers)
		for i := range a.LayerTypes {
			a.LayerTypes[i] = fullAttention
		}
	}
	theta, ok := ropeFloat(c.RopeParameters, "rope_theta")
	if !ok {
		return nil, fmt.Errorf("ministral3: rope_parameters.rope_theta is required")
	}
	a.RopeTheta = theta
	if beta, ok := ropeFloat(c.RopeParameters, "llama_4_scaling_beta"); ok {
		a.Llama4ScalingBeta = beta
	}
	if omp, ok := ropeFloat(c.RopeParameters, "original_max_position_embeddings"); ok {
		a.OriginalMaxPositionEmbeddings = int(omp)
	}
	if err := a.validate(); err != nil {
		return nil, err
	}
	return a, nil
}

// ropeFloat reads a numeric rope parameter; JSON numbers decode as float64.
func ropeFloat(m map[string]any, key string) (float64, bool) {
	v, ok := m[key]
	if !ok {
		return 0, false
	}
	f, ok := v.(float64)
	return f, ok
}

func (a *MinistralArgs) validate() error {
	switch {
	case a.HiddenSize <= 0:
		return fmt.Errorf("ministral3: hidden_size must be positive, got %d", a.HiddenSize)
	case a.HeadDim <= 0:
		return fmt.Errorf("ministral3: head_dim must be positive, got %d", a.HeadDim)
	case a.VocabSize <= 0:
		return fmt.Errorf("ministral3: vocab_size must be positive, got %d", a.VocabSize)
	case len(a.LayerTypes) == 0:
		return fmt.Errorf("ministral3: no layers (empty layer_types and num_hidden_layers)")
	case a.NumAttentionHeads%a.NumKeyValueHeads != 0:
		return fmt.Errorf("ministral3: num_attention_heads (%d) must be a multiple of num_key_value_heads (%d)",
			a.NumAttentionHeads, a.NumKeyValueHeads)
	}
	for i, t := range a.LayerTypes {
		if t != fullAttention && t != slidingAttention {
			return fmt.Errorf("ministral3: layer_types[%d] = %q, want %q or %q", i, t, fullAttention, slidingAttention)
		}
		if t == slidingAttention && a.SlidingWindow <= 0 {
			return fmt.Errorf("ministral3: sliding layer needs a positive sliding_window")
		}
	}
	return nil
}

// NumLayers is the effective decoder depth: the length of layer_types.
func (a *MinistralArgs) NumLayers() int { return len(a.LayerTypes) }

// Scale is the attention logit scale, head_dim raised to the -1/2 power.
func (a *MinistralArgs) Scale() float64 { return math.Pow(float64(a.HeadDim), -0.5) }

// QProjOut is the query projection output width.
func (a *MinistralArgs) QProjOut() int { return a.NumAttentionHeads * a.HeadDim }

// KVProjOut is the key (and value) projection output width.
func (a *MinistralArgs) KVProjOut() int { return a.NumKeyValueHeads * a.HeadDim }

// GQARepeat is the grouped-query repeat factor.
func (a *MinistralArgs) GQARepeat() int { return a.NumAttentionHeads / a.NumKeyValueHeads }

// IsSliding reports whether layer i uses sliding-window attention.
func (a *MinistralArgs) IsSliding(i int) bool { return a.LayerTypes[i] == slidingAttention }

// MakeCache builds one cache per layer: a rotating window cache for a sliding
// layer, a plain growing cache otherwise.
func (a *MinistralArgs) MakeCache() []compute.Cache {
	caches := make([]compute.Cache, a.NumLayers())
	for i := range caches {
		if a.IsSliding(i) {
			caches[i] = compute.NewRotatingKVCache(a.SlidingWindow, 0)
		} else {
			caches[i] = &compute.KVCache{}
		}
	}
	return caches
}

// AttnScale reproduces the llama4 query scale applied before attention:
// 1 + beta * log(1 + floor((arange(size) + offset) / max_position_embeddings)),
// one value per query position. With beta zero or all positions inside the first
// window the scale is uniformly one.
func (a *MinistralArgs) AttnScale(size, offset int) []float32 {
	out := make([]float32, size)
	beta := a.Llama4ScalingBeta
	maxPos := a.OriginalMaxPositionEmbeddings
	for i := range out {
		var floor float64
		if maxPos > 0 {
			floor = math.Floor(float64(i+offset) / float64(maxPos))
		}
		out[i] = float32(1 + beta*math.Log(1+floor))
	}
	return out
}

// AttnScaleBatch is the per-row llama4 query scale a left-padded ragged cohort
// needs: row b's queries sit at logical positions starting at offsets[b] (the
// cache's per-row RoPE offset, the padded cursor minus that row's left padding),
// so each row gets its own AttnScale. The result is the rows concatenated into a
// flat batch*size buffer, shaped [batch, 1, size, 1] by the caller to broadcast
// over heads and head_dim. A padding query (logical position before zero) lands
// in the masked region and its scale value is never read.
func (a *MinistralArgs) AttnScaleBatch(size int, offsets []int) []float32 {
	out := make([]float32, len(offsets)*size)
	for b, off := range offsets {
		copy(out[b*size:(b+1)*size], a.AttnScale(size, off))
	}
	return out
}

// WeightNames returns the sorted parameter key set: the four attention and three
// MLP projections (no bias) plus the two layernorms per layer, then the
// embedding, the final norm, and an untied lm_head.
func (a *MinistralArgs) WeightNames() []string {
	names := []string{"model.embed_tokens.weight"}
	for i := range a.NumLayers() {
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
	}
	names = append(names, "model.norm.weight")
	if !a.TieWordEmbeddings {
		names = append(names, "lm_head.weight")
	}
	sort.Strings(names)
	return names
}

// Sanitize drops the keys the model must not receive: the precomputed rotary
// inverse-frequency buffers, a tied checkpoint's stray lm_head.weight, and the
// per-tensor activation_scale buffers some fp8 checkpoints ship. The
// weight_scale_inv fusion (folding a block scale back into its weight) is a
// tensor multiply that belongs to the backend, so this surface leaves those keys
// in place for the loader to fuse.
func (a *MinistralArgs) Sanitize(weights map[string]*mlxgo.Array) map[string]*mlxgo.Array {
	for k := range weights {
		if strings.Contains(k, "self_attn.rotary_emb.inv_freq") || strings.Contains(k, "activation_scale") {
			delete(weights, k)
		}
	}
	if a.TieWordEmbeddings {
		delete(weights, "lm_head.weight")
	}
	return weights
}

// ministralLayer holds one decoder block's weights plus its attention kind.
type ministralLayer struct {
	sliding                bool
	inputLayernorm         *mlxgo.Array
	postAttentionLayernorm *mlxgo.Array
	qProj, kProj, vProj    *mlxgo.Array
	oProj                  *mlxgo.Array
	gateProj, upProj       *mlxgo.Array
	downProj               *mlxgo.Array
}

// Ministral3Model is an assembled Ministral 3 text model.
type Ministral3Model struct {
	args        *MinistralArgs
	embedTokens *mlxgo.Array
	layers      []ministralLayer
	norm        *mlxgo.Array
	lmHead      *mlxgo.Array // nil when tied
}

// NewMinistral3Model wires a sanitized weight map into a runnable model.
func NewMinistral3Model(args *MinistralArgs, weights map[string]*mlxgo.Array) (*Ministral3Model, error) {
	get := func(name string) (*mlxgo.Array, error) {
		w, ok := weights[name]
		if !ok || w == nil {
			return nil, fmt.Errorf("ministral3: missing weight %q", name)
		}
		return w, nil
	}
	m := &Ministral3Model{args: args, layers: make([]ministralLayer, args.NumLayers())}
	var err error
	if m.embedTokens, err = get("model.embed_tokens.weight"); err != nil {
		return nil, err
	}
	if m.norm, err = get("model.norm.weight"); err != nil {
		return nil, err
	}
	for i := range m.layers {
		m.layers[i].sliding = args.IsSliding(i)
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
	}
	if args.TieWordEmbeddings {
		m.lmHead = nil
	} else if m.lmHead, err = get("lm_head.weight"); err != nil {
		return nil, err
	}
	return m, nil
}

// Forward runs one sequence of tokens through the model and returns the logits,
// shaped [1, len(tokens), vocab_size]. It mirrors the Llama forward without the
// projection biases, with the llama4 query scale applied before attention. The
// sliding-window prefill mask and the rotating tensor cache for sliding layers
// are the remaining backend seam; a full-attention causal mask is requested here
// and the per-layer rotating bookkeeping bounds the window for decode.
func (m *Ministral3Model) Forward(tokens []int32, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	return m.forwardBL(tokens, 1, len(tokens), caches, s)
}

// BatchDecode runs one decode step for batch sequences at once and returns the
// logits, shaped [batch, 1, vocab_size]. tokens holds the batch's single tokens
// in row order, the [batch, 1] decode input the reference forms with
// inputs[:, None]. Every sequence shares the same cache length (a synchronized
// batch), so with L == 1 the step needs no mask and the [1, 1, 1, 1] query scale
// broadcasts across the batch.
func (m *Ministral3Model) BatchDecode(tokens []int32, batch int, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	return m.forwardBL(tokens, batch, 1, caches, s)
}

// forwardBL is the batch-polymorphic forward shared by Forward and BatchDecode.
// tokens is the row-major [batch, L] token matrix flattened to batch*L values
// and the result is [batch, L, vocab_size]; batch == 1 reproduces the
// single-sequence shapes and L == 1 is the batched decode step.
func (m *Ministral3Model) forwardBL(tokens []int32, batch, L int, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	if len(caches) != len(m.layers) {
		return nil, fmt.Errorf("ministral3: got %d caches, want %d", len(caches), len(m.layers))
	}
	a := m.args
	eps := float32(a.RMSNormEps)
	theta := float32(a.RopeTheta)
	scale := float32(a.Scale())
	hd := a.HeadDim
	nh := a.NumAttentionHeads
	nkv := a.NumKeyValueHeads

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

	mode, mask, err := caches[0].AttnMask(batch, L, s)
	if err != nil {
		return nil, err
	}
	ropeOff := caches[0].RopeOffsets()

	for i := range m.layers {
		layer := &m.layers[i]
		cache := caches[i]

		x := b.rmsNorm(h, layer.inputLayernorm, eps)
		q := b.linear(x, layer.qProj)
		k := b.linear(x, layer.kProj)
		v := b.linear(x, layer.vProj)
		q = b.transpose(b.reshape(q, []int{batch, L, nh, hd}), []int{0, 2, 1, 3})
		k = b.transpose(b.reshape(k, []int{batch, L, nkv, hd}), []int{0, 2, 1, 3})
		v = b.transpose(b.reshape(v, []int{batch, L, nkv, hd}), []int{0, 2, 1, 3})
		offset := cache.Offset
		if ropeOff == nil {
			q = b.rope(q, hd, theta, offset)
			k = b.rope(k, hd, theta, offset)
		} else {
			q = b.ropePerRow(q, ropeOff, func(r *mlxgo.Array, o int) *mlxgo.Array { return b.rope(r, hd, theta, o) })
			k = b.ropePerRow(k, ropeOff, func(r *mlxgo.Array, o int) *mlxgo.Array { return b.rope(r, hd, theta, o) })
		}
		// llama4 position-dependent query scale. A uniform cohort shares the offset
		// and broadcasts a [1, 1, L, 1] scale over batch; a left-padded cohort needs
		// a per-row [batch, 1, L, 1] because each row's queries sit at a different
		// logical position. Both broadcast over heads and head_dim.
		if b.err == nil {
			var as *mlxgo.Array
			var aerr error
			if ropeOff == nil {
				as, aerr = mlxgo.NewFloat32(a.AttnScale(L, offset), 1, 1, L, 1)
			} else {
				as, aerr = mlxgo.NewFloat32(a.AttnScaleBatch(L, ropeOff), batch, 1, L, 1)
			}
			if aerr != nil {
				b.err = aerr
			} else {
				q = b.mul(q, as)
			}
		}
		if b.err == nil {
			k, v, b.err = cache.Update(k, v, s)
		}
		attn := b.sdpaWith(q, k, v, scale, mode, mask)
		attn = b.reshape(b.transpose(attn, []int{0, 2, 1, 3}), []int{batch, L, nh * hd})
		attn = b.linear(attn, layer.oProj)
		h = b.add(h, attn)

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

// LoadMinistral3 assembles a runnable model from a checkpoint.
func LoadMinistral3(configJSON, blob []byte) (*Ministral3Model, error) {
	args, err := ParseMinistralArgs(configJSON)
	if err != nil {
		return nil, err
	}
	weights, err := compute.LoadTensors(blob)
	if err != nil {
		return nil, err
	}
	return NewMinistral3Model(args, args.Sanitize(weights))
}
