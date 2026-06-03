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

// Gemma4TextArgs decodes the Gemma 4 text config. It is the language model lifted
// out of the vision-language wrapper, and it is the most variant-rich decoder in
// the set. Four structural choices branch off the config:
//
//   - Layer kind. Each layer is sliding-window or full attention. layer_types is
//     explicit when present, otherwise it tiles the default pattern
//     (sliding_window_pattern-1 sliding layers then one full) up to the depth.
//   - KV sharing. The last num_kv_shared_layers layers do not own keys and values;
//     they read the cache of an earlier layer of the same kind. PreviousKVs maps
//     each layer to the layer whose keys and values it uses, and HasKV reports
//     which layers carry their own projections.
//   - Dual head dim. Full-attention layers use global_head_dim when it is set;
//     sliding layers use head_dim. The two can differ (512 vs 256 on 26B).
//   - K-eq-V. On the largest models full-attention layers reuse the keys as the
//     values, so those layers drop v_proj and may take a wider kv head count from
//     num_global_key_value_heads.
//
// The per-layer-input gating weights (embed_tokens_per_layer and friends) appear
// only when hidden_size_per_layer_input is positive, and a mixture-of-experts MLP
// replaces the dense MLP when enable_moe_block is set.
type Gemma4TextArgs struct {
	ModelType             string
	HiddenSize            int
	NumHiddenLayers       int
	IntermediateSize      int
	NumAttentionHeads     int
	HeadDim               int
	GlobalHeadDim         int
	RMSNormEps            float64
	VocabSize             int
	VocabSizePerLayerIn   int
	NumKeyValueHeads      int
	NumGlobalKVHeads      int // 0 means unset; falls back to NumKeyValueHeads
	NumKVSharedLayers     int
	HiddenSizePerLayerIn  int
	SlidingWindow         int
	SlidingWindowPattern  int
	MaxPositionEmbeddings int
	AttentionKEqV         bool
	FinalLogitSoftcapping float64
	EnableMoEBlock        bool
	TieWordEmbeddings     bool
	LayerTypes            []string

	// Per-attention-kind rotary parameters, keyed by layer kind.
	RopeTheta   map[string]float64
	RopePartial map[string]float64
}

type gemma4RopeParam struct {
	RopeTheta           *float64 `json:"rope_theta"`
	PartialRotaryFactor *float64 `json:"partial_rotary_factor"`
}

type gemma4Config struct {
	ModelType             string                     `json:"model_type"`
	HiddenSize            int                        `json:"hidden_size"`
	NumHiddenLayers       int                        `json:"num_hidden_layers"`
	IntermediateSize      int                        `json:"intermediate_size"`
	NumAttentionHeads     int                        `json:"num_attention_heads"`
	HeadDim               *int                       `json:"head_dim"`
	GlobalHeadDim         *int                       `json:"global_head_dim"`
	RMSNormEps            float64                    `json:"rms_norm_eps"`
	VocabSize             int                        `json:"vocab_size"`
	VocabSizePerLayerIn   *int                       `json:"vocab_size_per_layer_input"`
	NumKeyValueHeads      *int                       `json:"num_key_value_heads"`
	NumGlobalKVHeads      *int                       `json:"num_global_key_value_heads"`
	NumKVSharedLayers     int                        `json:"num_kv_shared_layers"`
	HiddenSizePerLayerIn  *int                       `json:"hidden_size_per_layer_input"`
	SlidingWindow         int                        `json:"sliding_window"`
	SlidingWindowPattern  int                        `json:"sliding_window_pattern"`
	MaxPositionEmbeddings int                        `json:"max_position_embeddings"`
	AttentionKEqV         bool                       `json:"attention_k_eq_v"`
	FinalLogitSoftcapping *float64                   `json:"final_logit_softcapping"`
	EnableMoEBlock        bool                       `json:"enable_moe_block"`
	TieWordEmbeddings     *bool                      `json:"tie_word_embeddings"`
	LayerTypes            []string                   `json:"layer_types"`
	RopeParameters        map[string]gemma4RopeParam `json:"rope_parameters"`
}

// Default rotary parameters when rope_parameters is absent, matching the upstream
// dataclass __post_init__.
var gemma4DefaultRope = map[string]gemma4RopeParam{
	fullAttention:    {RopeTheta: new(1000000.0), PartialRotaryFactor: new(0.25)},
	slidingAttention: {RopeTheta: new(10000.0), PartialRotaryFactor: new(1.0)},
}

// ParseGemma4TextArgs decodes a config.json body into Gemma4TextArgs, applying the
// dataclass defaults: head_dim 256, global_head_dim 512, num_key_value_heads 1,
// hidden_size_per_layer_input 256, final_logit_softcapping 30, tie_word_embeddings
// true, the layer_types default pattern, and the per-kind rotary defaults.
func ParseGemma4TextArgs(configJSON []byte) (*Gemma4TextArgs, error) {
	var c gemma4Config
	if err := json.Unmarshal(configJSON, &c); err != nil {
		return nil, fmt.Errorf("gemma4_text: decode config: %w", err)
	}
	if c.NumAttentionHeads <= 0 {
		return nil, fmt.Errorf("gemma4_text: num_attention_heads must be positive, got %d", c.NumAttentionHeads)
	}
	a := &Gemma4TextArgs{
		ModelType:             c.ModelType,
		HiddenSize:            c.HiddenSize,
		NumHiddenLayers:       c.NumHiddenLayers,
		IntermediateSize:      c.IntermediateSize,
		NumAttentionHeads:     c.NumAttentionHeads,
		RMSNormEps:            c.RMSNormEps,
		VocabSize:             c.VocabSize,
		NumKVSharedLayers:     c.NumKVSharedLayers,
		SlidingWindow:         c.SlidingWindow,
		SlidingWindowPattern:  c.SlidingWindowPattern,
		MaxPositionEmbeddings: c.MaxPositionEmbeddings,
		AttentionKEqV:         c.AttentionKEqV,
		EnableMoEBlock:        c.EnableMoEBlock,
	}
	a.HeadDim = intOr(c.HeadDim, 256)
	a.GlobalHeadDim = intOr(c.GlobalHeadDim, 512)
	a.NumKeyValueHeads = intOr(c.NumKeyValueHeads, 1)
	a.VocabSizePerLayerIn = intOr(c.VocabSizePerLayerIn, c.VocabSize)
	a.HiddenSizePerLayerIn = intOr(c.HiddenSizePerLayerIn, 256)
	if c.NumGlobalKVHeads != nil {
		a.NumGlobalKVHeads = *c.NumGlobalKVHeads
	}
	a.FinalLogitSoftcapping = 30.0
	if c.FinalLogitSoftcapping != nil {
		a.FinalLogitSoftcapping = *c.FinalLogitSoftcapping
	}
	a.TieWordEmbeddings = true
	if c.TieWordEmbeddings != nil {
		a.TieWordEmbeddings = *c.TieWordEmbeddings
	}
	if c.LayerTypes != nil {
		a.LayerTypes = c.LayerTypes
	} else {
		a.LayerTypes = gemma4DefaultLayerTypes(c.SlidingWindowPattern, c.NumHiddenLayers)
	}

	rope := c.RopeParameters
	if rope == nil {
		rope = gemma4DefaultRope
	}
	a.RopeTheta = map[string]float64{}
	a.RopePartial = map[string]float64{}
	for _, kind := range []string{fullAttention, slidingAttention} {
		p, ok := rope[kind]
		if !ok {
			p = gemma4DefaultRope[kind]
		}
		a.RopeTheta[kind] = floatOr(p.RopeTheta, 10000.0)
		a.RopePartial[kind] = floatOr(p.PartialRotaryFactor, 1.0)
	}
	if err := a.validate(); err != nil {
		return nil, err
	}
	return a, nil
}

func intOr(p *int, def int) int {
	if p != nil {
		return *p
	}
	return def
}

func floatOr(p *float64, def float64) float64 {
	if p != nil {
		return *p
	}
	return def
}

// gemma4DefaultLayerTypes tiles the default attention pattern to the requested
// depth: sliding_window_pattern-1 sliding layers followed by one full layer.
func gemma4DefaultLayerTypes(pattern, depth int) []string {
	if pattern < 1 {
		pattern = 1
	}
	base := make([]string, 0, pattern)
	for range pattern - 1 {
		base = append(base, slidingAttention)
	}
	base = append(base, fullAttention)
	out := make([]string, depth)
	for i := range out {
		out[i] = base[i%len(base)]
	}
	return out
}

func (a *Gemma4TextArgs) validate() error {
	switch {
	case a.HiddenSize <= 0:
		return fmt.Errorf("gemma4_text: hidden_size must be positive, got %d", a.HiddenSize)
	case a.HeadDim <= 0:
		return fmt.Errorf("gemma4_text: head_dim must be positive, got %d", a.HeadDim)
	case a.VocabSize <= 0:
		return fmt.Errorf("gemma4_text: vocab_size must be positive, got %d", a.VocabSize)
	case len(a.LayerTypes) == 0:
		return fmt.Errorf("gemma4_text: no layers (empty layer_types and num_hidden_layers)")
	case a.NumKVSharedLayers < 0 || a.NumKVSharedLayers > len(a.LayerTypes):
		return fmt.Errorf("gemma4_text: num_kv_shared_layers %d out of range for %d layers",
			a.NumKVSharedLayers, len(a.LayerTypes))
	}
	for i, t := range a.LayerTypes {
		if t != fullAttention && t != slidingAttention {
			return fmt.Errorf("gemma4_text: layer_types[%d] = %q, want %q or %q", i, t, fullAttention, slidingAttention)
		}
		if t == slidingAttention && a.SlidingWindow <= 0 {
			return fmt.Errorf("gemma4_text: sliding layer needs a positive sliding_window")
		}
	}
	return nil
}

// NumLayers is the decoder depth: the length of layer_types.
func (a *Gemma4TextArgs) NumLayers() int { return len(a.LayerTypes) }

// FirstKVShared is the index of the first KV-shared layer, equal to the number of
// layers that own keys and values: num_hidden_layers - num_kv_shared_layers.
func (a *Gemma4TextArgs) FirstKVShared() int { return a.NumLayers() - a.NumKVSharedLayers }

// HasKV reports whether layer i carries its own key and value projections. The
// last num_kv_shared_layers layers do not.
func (a *Gemma4TextArgs) HasKV(i int) bool { return i < a.FirstKVShared() }

// IsSliding reports whether layer i uses sliding-window attention.
func (a *Gemma4TextArgs) IsSliding(i int) bool { return a.LayerTypes[i] == slidingAttention }

// UseKEqV reports whether layer i reuses keys as values: enabled by
// attention_k_eq_v on full-attention layers only.
func (a *Gemma4TextArgs) UseKEqV(i int) bool { return a.AttentionKEqV && !a.IsSliding(i) }

// PreviousKVs maps each layer to the layer whose keys and values it reads. Layers
// that own their cache map to themselves; each KV-shared layer maps to the most
// recent owning layer of the same attention kind.
func (a *Gemma4TextArgs) PreviousKVs() []int {
	n := a.NumLayers()
	prev := make([]int, n)
	for i := range prev {
		prev[i] = i
	}
	m := a.FirstKVShared()
	if a.NumKVSharedLayers > 0 {
		byKind := map[string]int{}
		for i := range m {
			byKind[a.LayerTypes[i]] = i
		}
		for j := m; j < n; j++ {
			prev[j] = byKind[a.LayerTypes[j]]
		}
	}
	return prev
}

// PerLayerHeadDim is the per-head width of layer i: global_head_dim for a
// full-attention layer when it is set, otherwise head_dim.
func (a *Gemma4TextArgs) PerLayerHeadDim(i int) int {
	if !a.IsSliding(i) && a.GlobalHeadDim > 0 {
		return a.GlobalHeadDim
	}
	return a.HeadDim
}

// PerLayerNumKVHeads is the key/value head count of layer i: num_global_key_value_heads
// for a K-eq-V layer when it is set, otherwise num_key_value_heads.
func (a *Gemma4TextArgs) PerLayerNumKVHeads(i int) int {
	if a.UseKEqV(i) && a.NumGlobalKVHeads > 0 {
		return a.NumGlobalKVHeads
	}
	return a.NumKeyValueHeads
}

// LayerRopeTheta is the rotary base for layer i, selected by its attention kind.
func (a *Gemma4TextArgs) LayerRopeTheta(i int) float64 {
	if a.IsSliding(i) {
		return a.RopeTheta[slidingAttention]
	}
	return a.RopeTheta[fullAttention]
}

// LayerPartialRotary is the partial-rotary factor for layer i, selected by its
// attention kind. Only this fraction of each head's dimensions is rotated.
func (a *Gemma4TextArgs) LayerPartialRotary(i int) float64 {
	if a.IsSliding(i) {
		return a.RopePartial[slidingAttention]
	}
	return a.RopePartial[fullAttention]
}

// EmbedScale is the token-embedding multiplier, sqrt(hidden_size).
func (a *Gemma4TextArgs) EmbedScale() float64 { return math.Sqrt(float64(a.HiddenSize)) }

// HasPerLayerInputs reports whether the per-layer-input gating path is active,
// which holds when hidden_size_per_layer_input is positive.
func (a *Gemma4TextArgs) HasPerLayerInputs() bool { return a.HiddenSizePerLayerIn > 0 }

// PerLayerInputScales returns the three scalars of the per-layer-input gating
// path: the per-layer embedding scale sqrt(hidden_size_per_layer_input), the gate
// scale 2^-1/2, and the projection scale hidden_size^-1/2. They are meaningful
// only when HasPerLayerInputs is true.
func (a *Gemma4TextArgs) PerLayerInputScales() (embed, gate, projection float64) {
	embed = math.Sqrt(float64(a.HiddenSizePerLayerIn))
	gate = math.Pow(2, -0.5)
	projection = math.Pow(float64(a.HiddenSize), -0.5)
	return embed, gate, projection
}

// LogitSoftcap applies the final logit soft-cap to a value: tanh(x/cap)*cap. With
// a non-positive cap the value passes through unchanged.
func (a *Gemma4TextArgs) LogitSoftcap(x float64) float64 {
	return gemma4LogitSoftcap(a.FinalLogitSoftcapping, x)
}

func gemma4LogitSoftcap(cap, x float64) float64 {
	if cap <= 0 {
		return x
	}
	return math.Tanh(x/cap) * cap
}

// MakeCache builds one cache per owning layer (the first FirstKVShared layers): a
// plain growing cache for a full-attention layer, a rotating window cache for a
// sliding layer. KV-shared layers reuse these through PreviousKVs and so add no
// cache of their own.
func (a *Gemma4TextArgs) MakeCache() []compute.Cache {
	m := a.FirstKVShared()
	caches := make([]compute.Cache, m)
	for i := range m {
		if a.IsSliding(i) {
			caches[i] = compute.NewRotatingKVCache(a.SlidingWindow, 0)
		} else {
			caches[i] = &compute.KVCache{}
		}
	}
	return caches
}

// WeightNames returns the sorted parameter key set. Each layer carries the
// attention projections (q_proj plus, when it owns keys and values, k_proj and
// v_proj), the q and k norms (k_norm only on owning layers; v_norm is scale-free
// and carries no weight), the dense MLP, the four block layernorms, and a scalar
// layer_scalar. K-eq-V layers drop v_proj. When per-layer inputs are active each
// layer also carries the gate, projection, and projection norm of that path, and
// the model adds the per-layer embedding and projection tensors at the top level.
// An untied checkpoint adds lm_head.weight.
func (a *Gemma4TextArgs) WeightNames() []string {
	gated := a.HasPerLayerInputs()
	names := []string{
		"model.embed_tokens.weight",
		"model.norm.weight",
	}
	if gated {
		names = append(names,
			"model.embed_tokens_per_layer.weight",
			"model.per_layer_model_projection.weight",
			"model.per_layer_projection_norm.weight",
		)
	}
	for i := range a.NumLayers() {
		p := fmt.Sprintf("model.layers.%d.", i)
		names = append(names,
			p+"input_layernorm.weight",
			p+"layer_scalar",
			p+"mlp.down_proj.weight",
			p+"mlp.gate_proj.weight",
			p+"mlp.up_proj.weight",
			p+"post_attention_layernorm.weight",
			p+"post_feedforward_layernorm.weight",
			p+"pre_feedforward_layernorm.weight",
			p+"self_attn.o_proj.weight",
			p+"self_attn.q_norm.weight",
			p+"self_attn.q_proj.weight",
		)
		if a.HasKV(i) {
			names = append(names,
				p+"self_attn.k_norm.weight",
				p+"self_attn.k_proj.weight",
			)
			if !a.UseKEqV(i) {
				names = append(names, p+"self_attn.v_proj.weight")
			}
		}
		if gated {
			names = append(names,
				p+"per_layer_input_gate.weight",
				p+"per_layer_projection.weight",
				p+"post_per_layer_input_norm.weight",
			)
		}
	}
	if !a.TieWordEmbeddings {
		names = append(names, "lm_head.weight")
	}
	sort.Strings(names)
	return names
}

// Sanitize drops the keys the model must not receive: the precomputed rotary
// buffers and the per-tensor quant min/max buffers some checkpoints ship. The
// expert weight split (.experts.gate_up_proj into switch_glu gate and up halves,
// .experts.down_proj into switch_glu.down_proj) is a tensor split that belongs to
// the backend, so this surface leaves the MoE expert keys in place for the loader
// to split.
func (a *Gemma4TextArgs) Sanitize(weights map[string]*mlxgo.Array) map[string]*mlxgo.Array {
	for k := range weights {
		if strings.Contains(k, "self_attn.rotary_emb") ||
			strings.Contains(k, "input_max") || strings.Contains(k, "input_min") ||
			strings.Contains(k, "output_max") || strings.Contains(k, "output_min") {
			delete(weights, k)
		}
	}
	if a.TieWordEmbeddings {
		delete(weights, "lm_head.weight")
	}
	return weights
}
