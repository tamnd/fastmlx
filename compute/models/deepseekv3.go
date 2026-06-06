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

// DeepseekV3Args decodes the DeepSeek MLA plus mixture-of-experts config. This is
// the architecture the DeepSeek-V4 checkpoints ride on; the V4-specific pre-load
// patch is a weight rewrite layered on top of this surface. Two traits set it
// apart from every earlier family. The attention is multi-head latent attention:
// queries optionally pass through a low-rank q_a/q_b pair, keys and values are a
// single compressed latent (kv_lora_rank wide) plus a small rotary key band
// (qk_rope_head_dim), and the per-head key/value projections are absorbed into
// embed_q and unembed_out. The MLP is dense for the first first_k_dense_replace
// layers and then a routed mixture: a sigmoid gate scores the experts, a
// group-limited top-k picks them, and an optional shared expert runs alongside.
type DeepseekV3Args struct {
	ModelType             string
	VocabSize             int
	HiddenSize            int
	IntermediateSize      int
	MoEIntermediateSize   int
	NumHiddenLayers       int
	NumAttentionHeads     int
	NumKeyValueHeads      int
	NSharedExperts        int // 0 means none
	NRoutedExperts        int // 0 means a purely dense model
	RoutedScalingFactor   float64
	KVLoraRank            int
	QLoraRank             int // 0 means the plain q_proj path (no low-rank queries)
	QKRopeHeadDim         int
	VHeadDim              int
	QKNopeHeadDim         int
	TopkMethod            string
	ScoringFunc           string
	NormTopkProb          bool
	NGroup                int
	TopkGroup             int
	NumExpertsPerTok      int
	MoELayerFreq          int
	FirstKDenseReplace    int
	MaxPositionEmbeddings int
	RMSNormEps            float64
	RopeTheta             float64
	AttentionBias         bool
	RopeScaling           *DeepseekV3RopeScaling
}

// DeepseekV3RopeScaling carries the fields the attention scale and rope need.
type DeepseekV3RopeScaling struct {
	Type                          string
	Factor                        float64
	MScaleAllDim                  float64
	OriginalMaxPositionEmbeddings int
	BetaFast                      float64
	BetaSlow                      float64
}

type deepseekV3Config struct {
	ModelType             string          `json:"model_type"`
	VocabSize             int             `json:"vocab_size"`
	HiddenSize            int             `json:"hidden_size"`
	IntermediateSize      int             `json:"intermediate_size"`
	MoEIntermediateSize   int             `json:"moe_intermediate_size"`
	NumHiddenLayers       int             `json:"num_hidden_layers"`
	NumAttentionHeads     int             `json:"num_attention_heads"`
	NumKeyValueHeads      *int            `json:"num_key_value_heads"`
	NSharedExperts        *int            `json:"n_shared_experts"`
	NRoutedExperts        *int            `json:"n_routed_experts"`
	RoutedScalingFactor   *float64        `json:"routed_scaling_factor"`
	KVLoraRank            *int            `json:"kv_lora_rank"`
	QLoraRank             json.RawMessage `json:"q_lora_rank"`
	QKRopeHeadDim         *int            `json:"qk_rope_head_dim"`
	VHeadDim              *int            `json:"v_head_dim"`
	QKNopeHeadDim         *int            `json:"qk_nope_head_dim"`
	TopkMethod            *string         `json:"topk_method"`
	ScoringFunc           *string         `json:"scoring_func"`
	NormTopkProb          *bool           `json:"norm_topk_prob"`
	NGroup                *int            `json:"n_group"`
	TopkGroup             *int            `json:"topk_group"`
	NumExpertsPerTok      *int            `json:"num_experts_per_tok"`
	MoELayerFreq          *int            `json:"moe_layer_freq"`
	FirstKDenseReplace    *int            `json:"first_k_dense_replace"`
	MaxPositionEmbeddings *int            `json:"max_position_embeddings"`
	RMSNormEps            *float64        `json:"rms_norm_eps"`
	RopeTheta             *float64        `json:"rope_theta"`
	AttentionBias         *bool           `json:"attention_bias"`
	RopeScaling           json.RawMessage `json:"rope_scaling"`
}

type deepseekV3RopeScalingConfig struct {
	Type                          string  `json:"type"`
	RopeType                      string  `json:"rope_type"`
	Factor                        float64 `json:"factor"`
	MScaleAllDim                  float64 `json:"mscale_all_dim"`
	OriginalMaxPositionEmbeddings int     `json:"original_max_position_embeddings"`
	BetaFast                      float64 `json:"beta_fast"`
	BetaSlow                      float64 `json:"beta_slow"`
}

// ParseDeepseekV3Args decodes a config.json body into DeepseekV3Args, applying the
// dataclass defaults. The reference leaves q_lora_rank, n_shared_experts, and
// n_routed_experts as Optional fields: an absent q_lora_rank keeps the low-rank
// query default (1536), an explicit null selects the plain q_proj path, and an
// absent expert count means there is no mixture (a purely dense stack).
func ParseDeepseekV3Args(configJSON []byte) (*DeepseekV3Args, error) {
	var c deepseekV3Config
	if err := json.Unmarshal(configJSON, &c); err != nil {
		return nil, fmt.Errorf("deepseekv3: decode config: %w", err)
	}
	a := &DeepseekV3Args{
		ModelType:             c.ModelType,
		VocabSize:             c.VocabSize,
		HiddenSize:            c.HiddenSize,
		IntermediateSize:      c.IntermediateSize,
		MoEIntermediateSize:   c.MoEIntermediateSize,
		NumHiddenLayers:       c.NumHiddenLayers,
		NumAttentionHeads:     c.NumAttentionHeads,
		RoutedScalingFactor:   floatOr(c.RoutedScalingFactor, 1.0),
		KVLoraRank:            intOr(c.KVLoraRank, 512),
		QKRopeHeadDim:         intOr(c.QKRopeHeadDim, 64),
		VHeadDim:              intOr(c.VHeadDim, 128),
		QKNopeHeadDim:         intOr(c.QKNopeHeadDim, 128),
		NGroup:                intOr(c.NGroup, 1),
		TopkGroup:             intOr(c.TopkGroup, 1),
		NumExpertsPerTok:      intOr(c.NumExpertsPerTok, 1),
		MoELayerFreq:          intOr(c.MoELayerFreq, 1),
		FirstKDenseReplace:    intOr(c.FirstKDenseReplace, 0),
		MaxPositionEmbeddings: intOr(c.MaxPositionEmbeddings, 2048),
		RMSNormEps:            floatOr(c.RMSNormEps, 1e-6),
		RopeTheta:             floatOr(c.RopeTheta, 10000.0),
		NormTopkProb:          true,
		TopkMethod:            "noaux_tc",
		ScoringFunc:           "sigmoid",
	}
	a.NumKeyValueHeads = intOr(c.NumKeyValueHeads, c.NumAttentionHeads)
	a.NSharedExperts = intOr(c.NSharedExperts, 0)
	a.NRoutedExperts = intOr(c.NRoutedExperts, 0)
	if c.NormTopkProb != nil {
		a.NormTopkProb = *c.NormTopkProb
	}
	if c.TopkMethod != nil {
		a.TopkMethod = *c.TopkMethod
	}
	if c.ScoringFunc != nil {
		a.ScoringFunc = *c.ScoringFunc
	}
	if c.AttentionBias != nil {
		a.AttentionBias = *c.AttentionBias
	}
	// q_lora_rank: absent keeps the default 1536, an explicit null is the no-lora
	// path, and a number is taken as is.
	a.QLoraRank = 1536
	if len(c.QLoraRank) > 0 {
		if string(c.QLoraRank) == "null" {
			a.QLoraRank = 0
		} else {
			var v int
			if err := json.Unmarshal(c.QLoraRank, &v); err != nil {
				return nil, fmt.Errorf("deepseekv3: decode q_lora_rank: %w", err)
			}
			a.QLoraRank = v
		}
	}
	if err := a.resolveRopeScaling(c.RopeScaling); err != nil {
		return nil, err
	}
	if err := a.validate(); err != nil {
		return nil, err
	}
	return a, nil
}

func (a *DeepseekV3Args) resolveRopeScaling(raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var rs deepseekV3RopeScalingConfig
	if err := json.Unmarshal(raw, &rs); err != nil {
		return fmt.Errorf("deepseekv3: decode rope_scaling: %w", err)
	}
	kind := rs.Type
	if kind == "" {
		kind = rs.RopeType
	}
	a.RopeScaling = &DeepseekV3RopeScaling{
		Type:                          kind,
		Factor:                        rs.Factor,
		MScaleAllDim:                  rs.MScaleAllDim,
		OriginalMaxPositionEmbeddings: rs.OriginalMaxPositionEmbeddings,
		BetaFast:                      rs.BetaFast,
		BetaSlow:                      rs.BetaSlow,
	}
	return nil
}

func (a *DeepseekV3Args) validate() error {
	switch {
	case a.HiddenSize <= 0:
		return fmt.Errorf("deepseekv3: hidden_size must be positive, got %d", a.HiddenSize)
	case a.VocabSize <= 0:
		return fmt.Errorf("deepseekv3: vocab_size must be positive, got %d", a.VocabSize)
	case a.NumHiddenLayers <= 0:
		return fmt.Errorf("deepseekv3: num_hidden_layers must be positive, got %d", a.NumHiddenLayers)
	case a.NumAttentionHeads <= 0:
		return fmt.Errorf("deepseekv3: num_attention_heads must be positive, got %d", a.NumAttentionHeads)
	case a.QKNopeHeadDim <= 0 || a.QKRopeHeadDim <= 0 || a.VHeadDim <= 0:
		return fmt.Errorf("deepseekv3: head dims must be positive")
	case a.KVLoraRank <= 0:
		return fmt.Errorf("deepseekv3: kv_lora_rank must be positive, got %d", a.KVLoraRank)
	case a.TopkMethod != "noaux_tc":
		return fmt.Errorf("deepseekv3: unsupported topk_method %q (only noaux_tc)", a.TopkMethod)
	}
	if a.NRoutedExperts > 0 {
		switch {
		case a.NGroup <= 0:
			return fmt.Errorf("deepseekv3: n_group must be positive, got %d", a.NGroup)
		case a.NRoutedExperts%a.NGroup != 0:
			return fmt.Errorf("deepseekv3: n_routed_experts (%d) must be a multiple of n_group (%d)",
				a.NRoutedExperts, a.NGroup)
		case a.TopkGroup > a.NGroup:
			return fmt.Errorf("deepseekv3: topk_group (%d) exceeds n_group (%d)", a.TopkGroup, a.NGroup)
		case a.NumExpertsPerTok > a.NRoutedExperts:
			return fmt.Errorf("deepseekv3: num_experts_per_tok (%d) exceeds n_routed_experts (%d)",
				a.NumExpertsPerTok, a.NRoutedExperts)
		case a.NumExpertsPerTok <= 0:
			return fmt.Errorf("deepseekv3: num_experts_per_tok must be positive, got %d", a.NumExpertsPerTok)
		}
	}
	return nil
}

// NumLayers is the decoder depth.
func (a *DeepseekV3Args) NumLayers() int { return a.NumHiddenLayers }

// QHeadDim is the per-head query width, the nope plus rope bands.
func (a *DeepseekV3Args) QHeadDim() int { return a.QKNopeHeadDim + a.QKRopeHeadDim }

// HasQLora reports whether queries pass through the low-rank q_a/q_b pair.
func (a *DeepseekV3Args) HasQLora() bool { return a.QLoraRank > 0 }

// KVAOut is the width of kv_a_proj_with_mqa: the compressed latent plus the rope
// key band.
func (a *DeepseekV3Args) KVAOut() int { return a.KVLoraRank + a.QKRopeHeadDim }

// IsMoE reports whether the model has a routed mixture at all.
func (a *DeepseekV3Args) IsMoE() bool { return a.NRoutedExperts > 0 }

// HasSharedExperts reports whether a shared expert runs alongside the routed ones.
func (a *DeepseekV3Args) HasSharedExperts() bool { return a.NSharedExperts > 0 }

// AttentionScale is q_head_dim raised to the -1/2 power, adjusted by the yarn
// magnitude scale when rope_scaling carries a non-zero mscale_all_dim and the
// scaling factor grows the context: scale *= s*s with s = 0.1*mscale*log(factor)+1.
func (a *DeepseekV3Args) AttentionScale() float64 {
	scale := math.Pow(float64(a.QHeadDim()), -0.5)
	if a.RopeScaling != nil && a.RopeScaling.MScaleAllDim != 0 {
		factor := a.RopeScaling.Factor
		if factor > 1 {
			s := 0.1*a.RopeScaling.MScaleAllDim*math.Log(factor) + 1.0
			scale = scale * s * s
		}
	}
	return scale
}

// IsMoELayer reports whether layer idx uses the routed mixture: the model must
// have experts, the layer must be past the dense prefix, and it must fall on the
// mixture frequency.
func (a *DeepseekV3Args) IsMoELayer(idx int) bool {
	return a.IsMoE() && idx >= a.FirstKDenseReplace && idx%a.MoELayerFreq == 0
}

// LayerTypes returns "moe" or "dense" for each layer in order.
func (a *DeepseekV3Args) LayerTypes() []string {
	out := make([]string, a.NumLayers())
	for i := range out {
		if a.IsMoELayer(i) {
			out[i] = "moe"
		} else {
			out[i] = "dense"
		}
	}
	return out
}

// GroupExpertSelect reproduces the reference group_expert_select for one token
// row. It scores the experts with a sigmoid, adds the per-expert correction bias,
// restricts the choice to the topk_group highest-scoring groups (a group's score
// is the sum of its two best biased scores), takes the top_k experts by biased
// score, gathers the original (pre-bias) sigmoid scores at those experts,
// optionally normalizes them across the selection, and scales by
// routed_scaling_factor. The returned weights are aligned with the returned
// indices; order is not significant (the caller forms a weighted sum), so callers
// and tests should treat the pair as an expert-to-weight map.
func GroupExpertSelect(gates, bias []float32, topK, nGroup, topkGroup int, routedScalingFactor float64, normTopkProb bool) (inds []int, scores []float64) {
	n := len(gates)
	orig := make([]float64, n)
	biased := make([]float64, n)
	for i := range gates {
		s := 1.0 / (1.0 + math.Exp(-float64(gates[i])))
		orig[i] = s
		b := 0.0
		if i < len(bias) {
			b = float64(bias[i])
		}
		biased[i] = s + b
	}

	if nGroup > 1 {
		per := n / nGroup
		type gs struct {
			g     int
			score float64
		}
		groupScores := make([]gs, nGroup)
		for g := range nGroup {
			seg := append([]float64(nil), biased[g*per:(g+1)*per]...)
			sort.Sort(sort.Reverse(sort.Float64Slice(seg)))
			top := seg[0]
			if per > 1 {
				top += seg[1]
			}
			groupScores[g] = gs{g, top}
		}
		// Keep the topk_group groups with the highest score; zero the rest.
		sort.SliceStable(groupScores, func(i, j int) bool {
			return groupScores[i].score > groupScores[j].score
		})
		keep := make(map[int]bool, topkGroup)
		for i := 0; i < topkGroup && i < nGroup; i++ {
			keep[groupScores[i].g] = true
		}
		for g := range nGroup {
			if !keep[g] {
				for j := g * per; j < (g+1)*per; j++ {
					biased[j] = 0
				}
			}
		}
	}

	// Top-k experts by biased score.
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		return biased[order[i]] > biased[order[j]]
	})
	if topK > n {
		topK = n
	}
	inds = append(inds, order[:topK]...)

	scores = make([]float64, len(inds))
	for j, ix := range inds {
		scores[j] = orig[ix]
	}
	if topK > 1 && normTopkProb {
		denom := 0.0
		for _, s := range scores {
			denom += s
		}
		if denom != 0 {
			for j := range scores {
				scores[j] /= denom
			}
		}
	}
	for j := range scores {
		scores[j] *= routedScalingFactor
	}
	return inds, scores
}

// MakeCache builds one plain growing cache per layer. The latent attention stores
// the compressed kv and the rope key band as the cache's key and value tensors.
func (a *DeepseekV3Args) MakeCache() []*compute.KVCache {
	caches := make([]*compute.KVCache, a.NumLayers())
	for i := range caches {
		caches[i] = &compute.KVCache{}
	}
	return caches
}

// WeightNames returns the sorted parameter key set after the pre-load patch has
// run: the MLA projections (the q_a/q_b/layernorm trio or a plain q_proj, the
// kv_a projection and layernorm, the absorbed embed_q and unembed_out, and the
// output projection, with the optional attention biases), the two block
// layernorms, the per-layer MLP (dense gate/up/down, or the routed gate weight
// and correction bias, the stacked switch_mlp, and an optional shared expert),
// then the embedding, the final norm, and the head.
func (a *DeepseekV3Args) WeightNames() []string {
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
		)
		names = append(names, a.attnWeightNames(p+"self_attn.")...)
		names = append(names, a.mlpWeightNames(p+"mlp.", i)...)
	}
	sort.Strings(names)
	return names
}

func (a *DeepseekV3Args) attnWeightNames(p string) []string {
	var names []string
	if a.HasQLora() {
		names = append(names, p+"q_a_proj.weight", p+"q_a_layernorm.weight", p+"q_b_proj.weight")
		if a.AttentionBias {
			names = append(names, p+"q_a_proj.bias")
		}
	} else {
		names = append(names, p+"q_proj.weight")
	}
	names = append(names,
		p+"kv_a_proj_with_mqa.weight",
		p+"kv_a_layernorm.weight",
		p+"embed_q.weight",
		p+"unembed_out.weight",
		p+"o_proj.weight",
	)
	if a.AttentionBias {
		names = append(names, p+"kv_a_proj_with_mqa.bias", p+"o_proj.bias")
	}
	return names
}

func (a *DeepseekV3Args) mlpWeightNames(p string, idx int) []string {
	if !a.IsMoELayer(idx) {
		return []string{p + "gate_proj.weight", p + "up_proj.weight", p + "down_proj.weight"}
	}
	names := []string{
		p + "gate.weight",
		p + "gate.e_score_correction_bias",
		p + "switch_mlp.gate_proj.weight",
		p + "switch_mlp.up_proj.weight",
		p + "switch_mlp.down_proj.weight",
	}
	if a.HasSharedExperts() {
		names = append(names,
			p+"shared_experts.gate_proj.weight",
			p+"shared_experts.up_proj.weight",
			p+"shared_experts.down_proj.weight",
		)
	}
	return names
}

// Sanitize runs the host-only half of the pre-load patch: it drops the keys the
// model must not receive. The reference removes the multi-token-prediction layer
// (the one extra layer index past the decoder stack) and any precomputed rotary
// inverse-frequency buffers, both pure key filters. The backend half of the patch
// (fp8 dequant, the int4 remap, the per-expert stack, and the kv_b_proj split into
// embed_q and unembed_out) needs the GPU and lands in the constructor with the
// numeric forward, so stackExperts and the rest run there where an error can
// surface.
func (a *DeepseekV3Args) Sanitize(weights map[string]*mlxgo.Array) map[string]*mlxgo.Array {
	mtp := fmt.Sprintf("model.layers.%d.", a.NumLayers())
	out := make(map[string]*mlxgo.Array, len(weights))
	for k, v := range weights {
		if strings.HasPrefix(k, mtp) || strings.Contains(k, "rotary_emb.inv_freq") {
			continue
		}
		out[k] = v
	}
	return out
}

// remapInt4 rewrites a compressed-tensors int4 checkpoint into the affine triple
// the quantized kernels read, the "remap for int4" step of the reference sanitize.
// A quantized weight arrives as three keys: BASEweight_packed (the bit-packed
// values as uint8), BASEweight_scale (the per-group scale) and BASEweight_shape
// (the logical shape, which only marks the group). For each weight_shape it views
// the packed bytes back as uint32 (the dtype the quantized kernels index), keeps
// the scale as scales, derives biases as -8*scale (the int4 zero point folded into
// the affine bias), and drops the three source keys. The fp8 scale_inv keys end in
// a different suffix and are left for the fp8 dequant pass. A checkpoint with no
// weight_shape key passes through untouched. The view and the bias arithmetic are
// the backend ops; on the stub the first surfaces the unavailable error.
func remapInt4(weights map[string]*mlxgo.Array, s *mlxgo.Stream) error {
	var bases []string
	for k := range weights {
		if strings.HasSuffix(k, "weight_shape") {
			bases = append(bases, strings.TrimSuffix(k, "weight_shape"))
		}
	}
	for _, base := range bases {
		packed, ok := weights[base+"weight_packed"]
		if !ok {
			return fmt.Errorf("deepseekv3: int4 remap missing %q", base+"weight_packed")
		}
		scale, ok := weights[base+"weight_scale"]
		if !ok {
			return fmt.Errorf("deepseekv3: int4 remap missing %q", base+"weight_scale")
		}
		w, err := mlxgo.View(packed, mlxgo.Uint32, s)
		if err != nil {
			return err
		}
		biases, err := scaleNeg8(scale, s)
		if err != nil {
			return err
		}
		delete(weights, base+"weight_shape")
		delete(weights, base+"weight_packed")
		delete(weights, base+"weight_scale")
		weights[base+"weight"] = w
		weights[base+"scales"] = scale
		weights[base+"biases"] = biases
	}
	// The reference drops every weight_scale and weight_packed key, so clear any
	// orphan that carried no weight_shape sibling (fp8 weight_scale_inv keeps its
	// own suffix and is not matched here).
	for k := range weights {
		if strings.HasSuffix(k, "weight_scale") || strings.HasSuffix(k, "weight_packed") {
			delete(weights, k)
		}
	}
	return nil
}

// scaleNeg8 computes -8*scale in the scale's own dtype, the int4 affine bias. The
// product promotes to float32 against the scalar, so it is cast back to the scale
// dtype to match the reference, whose weak-typed -8*scale stays in the scale type.
func scaleNeg8(scale *mlxgo.Array, s *mlxgo.Stream) (*mlxgo.Array, error) {
	neg8, err := mlxgo.NewFloat32([]float32{-8}, 1)
	if err != nil {
		return nil, err
	}
	prod, err := mlxgo.Mul(scale, neg8, s)
	if err != nil {
		return nil, err
	}
	return mlxgo.Astype(prod, scale.Dtype(), s)
}

// stackExperts rewrites the per-expert MLP tensors a routed DeepSeek checkpoint
// carries (model.layers.L.mlp.experts.E.PROJ.COMP) into the stacked switch_mlp
// tensors the routed forward reads (model.layers.L.mlp.switch_mlp.PROJ.COMP), the
// "stack experts" step of the reference sanitize. PROJ is gate_proj, down_proj or
// up_proj; COMP is weight, and for a quantized checkpoint also scales and biases,
// so the stacked triple feeds the quantized switchGLU directly. For each present
// projection it joins the n_routed_experts per-expert arrays along a new leading
// expert axis and pops the per-expert keys. A checkpoint that already carries the
// stacked switch_mlp tensors has no per-expert key, so it passes through untouched.
// The stack is the one backend op; on the stub it surfaces the unavailable error.
func stackExperts(weights map[string]*mlxgo.Array, numLayers, numExperts int, s *mlxgo.Stream) error {
	for l := 0; l < numLayers; l++ {
		for _, proj := range [...]string{"gate_proj", "down_proj", "up_proj"} {
			for _, comp := range [...]string{"weight", "scales", "biases"} {
				if _, ok := weights[fmt.Sprintf("model.layers.%d.mlp.experts.0.%s.%s", l, proj, comp)]; !ok {
					continue
				}
				parts := make([]*mlxgo.Array, numExperts)
				for e := 0; e < numExperts; e++ {
					key := fmt.Sprintf("model.layers.%d.mlp.experts.%d.%s.%s", l, e, proj, comp)
					a, ok := weights[key]
					if !ok {
						return fmt.Errorf("deepseekv3: stack experts missing %q", key)
					}
					parts[e] = a
				}
				stacked, err := mlxgo.Stack(parts, 0, s)
				if err != nil {
					return err
				}
				for e := 0; e < numExperts; e++ {
					delete(weights, fmt.Sprintf("model.layers.%d.mlp.experts.%d.%s.%s", l, e, proj, comp))
				}
				weights[fmt.Sprintf("model.layers.%d.mlp.switch_mlp.%s.%s", l, proj, comp)] = stacked
			}
		}
	}
	return nil
}
