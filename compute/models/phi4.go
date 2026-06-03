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

// Phi4Args decodes the phi3 config that the Phi-4 family rides on. It is a
// Llama-shaped decoder with two distinguishing traits. The attention projection
// is fused: one qkv_proj emits the query, key, and value bands back to back
// (width num_heads*head_dim + 2*num_kv_heads*head_dim), split at run time at the
// two band boundaries. The rotary embedding is partial (only the first
// head_dim*partial_rotary dims rotate) and selectable: a longrope or su scaling
// switches to the SuScaledRoPE long-context frequencies, a linear scaling shrinks
// positions by 1/factor, and anything else is plain rope. The MLP fuses gate and
// up like GLM. The model is untied unless tie_word_embeddings is set.
type Phi4Args struct {
	ModelType                     string
	HiddenSize                    int
	NumHiddenLayers               int
	IntermediateSize              int
	NumAttentionHeads             int
	RMSNormEps                    float64
	VocabSize                     int
	NumKeyValueHeads              int
	RopeTheta                     float64
	RopeTraditional               bool
	PartialRotaryFactor           float64
	MaxPositionEmbeddings         int
	OriginalMaxPositionEmbeddings int
	TieWordEmbeddings             bool
	// RopeScaling is the validated scaling, or nil when absent or disabled by the
	// reference's __post_init__ (an unsupported type is dropped with a warning).
	RopeScaling *Phi4RopeScaling
}

// Phi4RopeScaling is the resolved rope_scaling block.
type Phi4RopeScaling struct {
	Type                          string
	Factor                        float64
	ShortFactor                   []float64
	LongFactor                    []float64
	OriginalMaxPositionEmbeddings int
}

type phi4Config struct {
	ModelType                     string          `json:"model_type"`
	HiddenSize                    int             `json:"hidden_size"`
	NumHiddenLayers               int             `json:"num_hidden_layers"`
	IntermediateSize              int             `json:"intermediate_size"`
	NumAttentionHeads             int             `json:"num_attention_heads"`
	RMSNormEps                    float64         `json:"rms_norm_eps"`
	VocabSize                     int             `json:"vocab_size"`
	NumKeyValueHeads              *int            `json:"num_key_value_heads"`
	RopeTheta                     *float64        `json:"rope_theta"`
	RopeTraditional               *bool           `json:"rope_traditional"`
	PartialRotaryFactor           *float64        `json:"partial_rotary_factor"`
	MaxPositionEmbeddings         *int            `json:"max_position_embeddings"`
	OriginalMaxPositionEmbeddings *int            `json:"original_max_position_embeddings"`
	TieWordEmbeddings             *bool           `json:"tie_word_embeddings"`
	RopeScaling                   json.RawMessage `json:"rope_scaling"`
}

type phi4RopeScalingConfig struct {
	Type                          string    `json:"type"`
	RopeType                      string    `json:"rope_type"`
	Factor                        float64   `json:"factor"`
	ShortFactor                   []float64 `json:"short_factor"`
	LongFactor                    []float64 `json:"long_factor"`
	OriginalMaxPositionEmbeddings int       `json:"original_max_position_embeddings"`
}

// ParsePhi4Args decodes a config.json body into Phi4Args, applying the dataclass
// defaults (num_key_value_heads falls back to num_attention_heads, rope_theta to
// 10000, partial_rotary_factor to 1.0, max_position_embeddings to 131072,
// original_max_position_embeddings to 4096) and reproducing the reference's
// rope_scaling validation: a scaling block must carry both "long_factor" and
// "type", and a type outside longrope/su/linear is dropped (the reference prints
// a warning and continues with no scaling).
func ParsePhi4Args(configJSON []byte) (*Phi4Args, error) {
	var c phi4Config
	if err := json.Unmarshal(configJSON, &c); err != nil {
		return nil, fmt.Errorf("phi4: decode config: %w", err)
	}
	if c.NumAttentionHeads <= 0 {
		return nil, fmt.Errorf("phi4: num_attention_heads must be positive, got %d", c.NumAttentionHeads)
	}
	a := &Phi4Args{
		ModelType:                     c.ModelType,
		HiddenSize:                    c.HiddenSize,
		NumHiddenLayers:               c.NumHiddenLayers,
		IntermediateSize:              c.IntermediateSize,
		NumAttentionHeads:             c.NumAttentionHeads,
		RMSNormEps:                    c.RMSNormEps,
		VocabSize:                     c.VocabSize,
		RopeTheta:                     10000,
		PartialRotaryFactor:           1.0,
		MaxPositionEmbeddings:         131072,
		OriginalMaxPositionEmbeddings: 4096,
	}
	a.NumKeyValueHeads = c.NumAttentionHeads
	if c.NumKeyValueHeads != nil {
		a.NumKeyValueHeads = *c.NumKeyValueHeads
	}
	if c.RopeTheta != nil {
		a.RopeTheta = *c.RopeTheta
	}
	if c.RopeTraditional != nil {
		a.RopeTraditional = *c.RopeTraditional
	}
	if c.PartialRotaryFactor != nil {
		a.PartialRotaryFactor = *c.PartialRotaryFactor
	}
	if c.MaxPositionEmbeddings != nil {
		a.MaxPositionEmbeddings = *c.MaxPositionEmbeddings
	}
	if c.OriginalMaxPositionEmbeddings != nil {
		a.OriginalMaxPositionEmbeddings = *c.OriginalMaxPositionEmbeddings
	}
	if c.TieWordEmbeddings != nil {
		a.TieWordEmbeddings = *c.TieWordEmbeddings
	}
	if err := a.resolveRopeScaling(c.RopeScaling); err != nil {
		return nil, err
	}
	if err := a.validate(); err != nil {
		return nil, err
	}
	return a, nil
}

// resolveRopeScaling reproduces the reference __post_init__: a present scaling
// block must contain "long_factor" and "type", and a type that is not one of
// longrope/su/linear is dropped (left nil).
func (a *Phi4Args) resolveRopeScaling(raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(raw, &keys); err != nil {
		return fmt.Errorf("phi4: decode rope_scaling: %w", err)
	}
	if _, ok := keys["long_factor"]; !ok {
		return fmt.Errorf("phi4: rope_scaling must contain keys {long_factor, type}")
	}
	if _, ok := keys["type"]; !ok {
		return fmt.Errorf("phi4: rope_scaling must contain keys {long_factor, type}")
	}
	var rs phi4RopeScalingConfig
	if err := json.Unmarshal(raw, &rs); err != nil {
		return fmt.Errorf("phi4: decode rope_scaling: %w", err)
	}
	kind := rs.Type
	switch kind {
	case "longrope", "su", "linear":
	default:
		// Unsupported type: the reference warns and continues unscaled.
		return nil
	}
	orig := rs.OriginalMaxPositionEmbeddings
	if orig == 0 {
		orig = a.OriginalMaxPositionEmbeddings
	}
	a.RopeScaling = &Phi4RopeScaling{
		Type:                          kind,
		Factor:                        rs.Factor,
		ShortFactor:                   rs.ShortFactor,
		LongFactor:                    rs.LongFactor,
		OriginalMaxPositionEmbeddings: orig,
	}
	return nil
}

func (a *Phi4Args) validate() error {
	switch {
	case a.HiddenSize <= 0:
		return fmt.Errorf("phi4: hidden_size must be positive, got %d", a.HiddenSize)
	case a.VocabSize <= 0:
		return fmt.Errorf("phi4: vocab_size must be positive, got %d", a.VocabSize)
	case a.NumHiddenLayers <= 0:
		return fmt.Errorf("phi4: num_hidden_layers must be positive, got %d", a.NumHiddenLayers)
	case a.NumKeyValueHeads <= 0:
		return fmt.Errorf("phi4: num_key_value_heads must be positive, got %d", a.NumKeyValueHeads)
	case a.HiddenSize%a.NumAttentionHeads != 0:
		return fmt.Errorf("phi4: hidden_size (%d) must be a multiple of num_attention_heads (%d)",
			a.HiddenSize, a.NumAttentionHeads)
	case a.NumAttentionHeads%a.NumKeyValueHeads != 0:
		return fmt.Errorf("phi4: num_attention_heads (%d) must be a multiple of num_key_value_heads (%d)",
			a.NumAttentionHeads, a.NumKeyValueHeads)
	case a.PartialRotaryFactor <= 0 || a.PartialRotaryFactor > 1:
		return fmt.Errorf("phi4: partial_rotary_factor must be in (0,1], got %g", a.PartialRotaryFactor)
	}
	return nil
}

// NumLayers is the decoder depth.
func (a *Phi4Args) NumLayers() int { return a.NumHiddenLayers }

// HeadDim is hidden_size / num_attention_heads (phi3 carries no head_dim field).
func (a *Phi4Args) HeadDim() int { return a.HiddenSize / a.NumAttentionHeads }

// Scale is the attention logit scale, head_dim raised to the -1/2 power.
func (a *Phi4Args) Scale() float64 { return math.Pow(float64(a.HeadDim()), -0.5) }

// QueryPos is the width of the query band in the fused qkv projection.
func (a *Phi4Args) QueryPos() int { return a.NumAttentionHeads * a.HeadDim() }

// KVSize is the width of each of the key and value bands.
func (a *Phi4Args) KVSize() int { return a.NumKeyValueHeads * a.HeadDim() }

// OpSize is the fused qkv projection output width: one query band plus two kv bands.
func (a *Phi4Args) OpSize() int { return a.QueryPos() + 2*a.KVSize() }

// RopeDims is the number of head dimensions the rotary embedding rotates:
// floor(head_dim * partial_rotary_factor), matching the reference int() truncation.
func (a *Phi4Args) RopeDims() int { return int(float64(a.HeadDim()) * a.PartialRotaryFactor) }

// UsesSuRope reports whether the resolved scaling selects the SuScaledRoPE
// long-context path (longrope or su).
func (a *Phi4Args) UsesSuRope() bool {
	return a.RopeScaling != nil && (a.RopeScaling.Type == "longrope" || a.RopeScaling.Type == "su")
}

// LinearRopeScale is 1/factor when a linear scaling is active, else 1.0.
func (a *Phi4Args) LinearRopeScale() float64 {
	if a.RopeScaling != nil && a.RopeScaling.Type == "linear" {
		return 1.0 / a.RopeScaling.Factor
	}
	return 1.0
}

// SuRopeFreqs returns the SuScaledRoPE frequencies, long_factor[i] times
// base**(2i/dims) over the rotated dimensions, matching the reference
// `mx.array(long_factor) * base ** (arange(0, dims, 2) / dims)`.
func (a *Phi4Args) SuRopeFreqs() []float64 {
	rs := a.RopeScaling
	dims := a.RopeDims()
	n := dims / 2
	out := make([]float64, n)
	for i := range out {
		freq := math.Pow(a.RopeTheta, float64(2*i)/float64(dims))
		f := 1.0
		if i < len(rs.LongFactor) {
			f = rs.LongFactor[i]
		}
		out[i] = f * freq
	}
	return out
}

// SuRopeScale is the SuScaledRoPE magnitude scale applied to the rotated dims
// before the embedding: 1 when the context does not grow, else
// sqrt(1 + log(factor)/log(original_max_position_embeddings)).
func (a *Phi4Args) SuRopeScale() float64 {
	orig := a.RopeScaling.OriginalMaxPositionEmbeddings
	factor := float64(a.MaxPositionEmbeddings) / float64(orig)
	if factor <= 1.0 {
		return 1.0
	}
	return math.Sqrt(1 + math.Log(factor)/math.Log(float64(orig)))
}

// MakeCache builds one plain growing cache per layer.
func (a *Phi4Args) MakeCache() []*compute.KVCache {
	caches := make([]*compute.KVCache, a.NumLayers())
	for i := range caches {
		caches[i] = &compute.KVCache{}
	}
	return caches
}

// WeightNames returns the sorted parameter key set: the fused qkv and the output
// projection, the fused gate-up and down projections, the two block layernorms,
// then the embedding, the final norm, and lm_head when the head is untied.
func (a *Phi4Args) WeightNames() []string {
	names := []string{
		"model.embed_tokens.weight",
		"model.norm.weight",
	}
	if !a.TieWordEmbeddings {
		names = append(names, "lm_head.weight")
	}
	for i := range a.NumLayers() {
		p := fmt.Sprintf("model.layers.%d.", i)
		names = append(names,
			p+"input_layernorm.weight",
			p+"post_attention_layernorm.weight",
			p+"self_attn.qkv_proj.weight",
			p+"self_attn.o_proj.weight",
			p+"mlp.gate_up_proj.weight",
			p+"mlp.down_proj.weight",
		)
	}
	sort.Strings(names)
	return names
}

// Sanitize is the identity: the reference phi3 model defines no weight rewrite.
func (a *Phi4Args) Sanitize(weights map[string]*mlxgo.Array) map[string]*mlxgo.Array {
	return weights
}

// phi4Layer holds one decoder block's weights.
type phi4Layer struct {
	inputLayernorm         *mlxgo.Array
	postAttentionLayernorm *mlxgo.Array
	qkvProj                *mlxgo.Array
	oProj                  *mlxgo.Array
	gateUpProj             *mlxgo.Array
	downProj               *mlxgo.Array
}

// Phi4Model is an assembled phi3-family model.
type Phi4Model struct {
	args        *Phi4Args
	embedTokens *mlxgo.Array
	layers      []phi4Layer
	norm        *mlxgo.Array
	lmHead      *mlxgo.Array // nil when tied
}

// NewPhi4Model wires a sanitized weight map into a runnable model.
func NewPhi4Model(args *Phi4Args, weights map[string]*mlxgo.Array) (*Phi4Model, error) {
	get := func(name string) (*mlxgo.Array, error) {
		w, ok := weights[name]
		if !ok || w == nil {
			return nil, fmt.Errorf("phi4: missing weight %q", name)
		}
		return w, nil
	}
	m := &Phi4Model{args: args, layers: make([]phi4Layer, args.NumLayers())}
	var err error
	if m.embedTokens, err = get("model.embed_tokens.weight"); err != nil {
		return nil, err
	}
	if m.norm, err = get("model.norm.weight"); err != nil {
		return nil, err
	}
	if !args.TieWordEmbeddings {
		if m.lmHead, err = get("lm_head.weight"); err != nil {
			return nil, err
		}
	}
	for i := range m.layers {
		p := fmt.Sprintf("model.layers.%d.", i)
		req := []struct {
			name string
			dst  **mlxgo.Array
		}{
			{p + "input_layernorm.weight", &m.layers[i].inputLayernorm},
			{p + "post_attention_layernorm.weight", &m.layers[i].postAttentionLayernorm},
			{p + "self_attn.qkv_proj.weight", &m.layers[i].qkvProj},
			{p + "self_attn.o_proj.weight", &m.layers[i].oProj},
			{p + "mlp.gate_up_proj.weight", &m.layers[i].gateUpProj},
			{p + "mlp.down_proj.weight", &m.layers[i].downProj},
		}
		for _, f := range req {
			if *f.dst, err = get(f.name); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}

// Forward runs one sequence of tokens through the model and returns the logits,
// shaped [1, len(tokens), vocab_size]. It mirrors the phi3 block: a standard
// residual around fused-qkv attention, then a standard residual around the fused
// gate-up SwiGLU MLP. The fused qkv projection is split at run time into the
// query, key, and value bands. The longrope/su SuScaledRoPE path is staged behind
// a seam; the standard and linear rope paths run here.
func (m *Phi4Model) Forward(tokens []int32, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	if len(caches) != len(m.layers) {
		return nil, fmt.Errorf("phi4: got %d caches, want %d", len(caches), len(m.layers))
	}
	a := m.args
	if a.UsesSuRope() {
		return nil, fmt.Errorf("phi4: SuScaledRoPE (longrope/su) forward is staged behind a seam")
	}
	L := len(tokens)
	eps := float32(a.RMSNormEps)
	theta := float32(a.RopeTheta)
	scale := float32(a.Scale())
	ropeScale := float32(a.LinearRopeScale())
	hd := a.HeadDim()
	nh := a.NumAttentionHeads
	nkv := a.NumKeyValueHeads
	ropeDims := a.RopeDims()
	trad := a.RopeTraditional
	qpos := a.QueryPos()
	kvsz := a.KVSize()

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
		qkv := b.linear(x, layer.qkvProj)
		q, k, v := b.splitQKV(qkv, qpos, kvsz)
		q = b.transpose(b.reshape(q, []int{1, L, nh, hd}), []int{0, 2, 1, 3})
		k = b.transpose(b.reshape(k, []int{1, L, nkv, hd}), []int{0, 2, 1, 3})
		v = b.transpose(b.reshape(v, []int{1, L, nkv, hd}), []int{0, 2, 1, 3})
		offset := cache.Offset
		q = b.ropeScaled(q, ropeDims, trad, theta, ropeScale, offset)
		k = b.ropeScaled(k, ropeDims, trad, theta, ropeScale, offset)
		if b.err == nil {
			k, v, b.err = cache.Update(k, v, s)
		}
		attn := b.sdpa(q, k, v, scale, maskMode)
		attn = b.reshape(b.transpose(attn, []int{0, 2, 1, 3}), []int{1, L, nh * hd})
		attn = b.linear(attn, layer.oProj)
		h = b.add(h, attn)

		y := b.rmsNorm(h, layer.postAttentionLayernorm, eps)
		mlp := b.phiMLP(y, layer)
		h = b.add(h, mlp)
	}

	h = b.rmsNorm(h, m.norm, eps)
	var logits *mlxgo.Array
	if m.lmHead != nil {
		logits = b.linear(h, m.lmHead)
	} else {
		logits = b.linear(h, m.embedTokens)
	}
	if b.err != nil {
		return nil, b.err
	}
	return logits, nil
}

// splitQKV carves the fused qkv projection into the query, key, and value bands
// at the two band boundaries.
func (b *fb) splitQKV(qkv *mlxgo.Array, queryPos, kvSize int) (q, k, v *mlxgo.Array) {
	if b.err != nil {
		return nil, nil, nil
	}
	parts, err := mlxgo.SplitSections(qkv, []int{queryPos, queryPos + kvSize}, -1, b.s)
	if err != nil {
		b.err = err
		return nil, nil, nil
	}
	return parts[0], parts[1], parts[2]
}

// phiMLP runs the fused gate-up projection, splits it into the gate and up halves,
// and combines them through SwiGLU before the down projection.
func (b *fb) phiMLP(x *mlxgo.Array, layer *phi4Layer) *mlxgo.Array {
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

// LoadPhi4 assembles a runnable model from a checkpoint.
func LoadPhi4(configJSON, blob []byte) (*Phi4Model, error) {
	args, err := ParsePhi4Args(configJSON)
	if err != nil {
		return nil, err
	}
	weights, err := compute.LoadTensors(blob)
	if err != nil {
		return nil, err
	}
	return NewPhi4Model(args, args.Sanitize(weights))
}
