// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"sort"

	"github.com/tamnd/fastmlx/compute"
	"github.com/tamnd/fastmlx/mlxgo"
)

// Qwen3NextArgs decodes the Qwen3-Next hybrid config. This family interleaves two
// kinds of token mixer down the stack: most layers are a gated delta net (a
// linear, recurrent mixer with a depthwise causal convolution and a per-head
// recurrent state), and every full_attention_interval-th layer is ordinary
// scaled-dot-product attention with a gated output. The MLP is either a dense
// SwiGLU or a sparse mixture with a shared expert, chosen per layer. The two
// mixer kinds need two different caches, and the recurrent one is not trimmable,
// which is the trait that shapes prefix caching for the family.
type Qwen3NextArgs struct {
	ModelType                    string
	HiddenSize                   int
	NumHiddenLayers              int
	IntermediateSize             int
	NumAttentionHeads            int
	LinearNumValueHeads          int
	LinearNumKeyHeads            int
	LinearKeyHeadDim             int
	LinearValueHeadDim           int
	LinearConvKernelDim          int
	NumExperts                   int
	NumExpertsPerTok             int
	DecoderSparseStep            int
	SharedExpertIntermediateSize int
	MLPOnlyLayers                []int
	MoEIntermediateSize          int
	RMSNormEps                   float64
	VocabSize                    int
	NumKeyValueHeads             int
	RopeTheta                    float64
	PartialRotaryFactor          float64
	MaxPositionEmbeddings        int
	HeadDim                      int
	NormTopkProb                 bool
	TieWordEmbeddings            bool
	AttentionBias                bool
	FullAttentionInterval        int
	RopeScaling                  map[string]any
	quant                        quantConfig
}

type qwen3NextConfig struct {
	ModelType                    string         `json:"model_type"`
	HiddenSize                   int            `json:"hidden_size"`
	NumHiddenLayers              int            `json:"num_hidden_layers"`
	IntermediateSize             int            `json:"intermediate_size"`
	NumAttentionHeads            int            `json:"num_attention_heads"`
	LinearNumValueHeads          int            `json:"linear_num_value_heads"`
	LinearNumKeyHeads            int            `json:"linear_num_key_heads"`
	LinearKeyHeadDim             int            `json:"linear_key_head_dim"`
	LinearValueHeadDim           int            `json:"linear_value_head_dim"`
	LinearConvKernelDim          int            `json:"linear_conv_kernel_dim"`
	NumExperts                   int            `json:"num_experts"`
	NumExpertsPerTok             int            `json:"num_experts_per_tok"`
	DecoderSparseStep            int            `json:"decoder_sparse_step"`
	SharedExpertIntermediateSize int            `json:"shared_expert_intermediate_size"`
	MLPOnlyLayers                []int          `json:"mlp_only_layers"`
	MoEIntermediateSize          int            `json:"moe_intermediate_size"`
	RMSNormEps                   *float64       `json:"rms_norm_eps"`
	VocabSize                    int            `json:"vocab_size"`
	NumKeyValueHeads             int            `json:"num_key_value_heads"`
	RopeTheta                    *float64       `json:"rope_theta"`
	PartialRotaryFactor          *float64       `json:"partial_rotary_factor"`
	MaxPositionEmbeddings        int            `json:"max_position_embeddings"`
	HeadDim                      int            `json:"head_dim"`
	NormTopkProb                 *bool          `json:"norm_topk_prob"`
	TieWordEmbeddings            *bool          `json:"tie_word_embeddings"`
	AttentionBias                *bool          `json:"attention_bias"`
	FullAttentionInterval        *int           `json:"full_attention_interval"`
	RopeScaling                  map[string]any `json:"rope_scaling"`
}

// ParseQwen3NextArgs decodes a config.json body into Qwen3NextArgs and applies the
// dataclass defaults (norm_topk_prob false, untied, no attention bias, a full
// attention every fourth layer, rope_theta 10000, partial_rotary_factor 1.0).
func ParseQwen3NextArgs(configJSON []byte) (*Qwen3NextArgs, error) {
	var c qwen3NextConfig
	if err := json.Unmarshal(configJSON, &c); err != nil {
		return nil, fmt.Errorf("qwen3next: decode config: %w", err)
	}
	a := &Qwen3NextArgs{
		ModelType:                    c.ModelType,
		HiddenSize:                   c.HiddenSize,
		NumHiddenLayers:              c.NumHiddenLayers,
		IntermediateSize:             c.IntermediateSize,
		NumAttentionHeads:            c.NumAttentionHeads,
		LinearNumValueHeads:          c.LinearNumValueHeads,
		LinearNumKeyHeads:            c.LinearNumKeyHeads,
		LinearKeyHeadDim:             c.LinearKeyHeadDim,
		LinearValueHeadDim:           c.LinearValueHeadDim,
		LinearConvKernelDim:          c.LinearConvKernelDim,
		NumExperts:                   c.NumExperts,
		NumExpertsPerTok:             c.NumExpertsPerTok,
		DecoderSparseStep:            c.DecoderSparseStep,
		SharedExpertIntermediateSize: c.SharedExpertIntermediateSize,
		MLPOnlyLayers:                c.MLPOnlyLayers,
		MoEIntermediateSize:          c.MoEIntermediateSize,
		RMSNormEps:                   floatOr(c.RMSNormEps, 1e-6),
		VocabSize:                    c.VocabSize,
		NumKeyValueHeads:             c.NumKeyValueHeads,
		RopeTheta:                    floatOr(c.RopeTheta, 10000.0),
		PartialRotaryFactor:          floatOr(c.PartialRotaryFactor, 1.0),
		MaxPositionEmbeddings:        c.MaxPositionEmbeddings,
		HeadDim:                      c.HeadDim,
		FullAttentionInterval:        intOr(c.FullAttentionInterval, 4),
		RopeScaling:                  c.RopeScaling,
	}
	if c.NormTopkProb != nil {
		a.NormTopkProb = *c.NormTopkProb
	}
	if c.TieWordEmbeddings != nil {
		a.TieWordEmbeddings = *c.TieWordEmbeddings
	}
	if c.AttentionBias != nil {
		a.AttentionBias = *c.AttentionBias
	}
	if a.MLPOnlyLayers == nil {
		a.MLPOnlyLayers = []int{}
	}
	quant, err := parseQuantConfig(configJSON)
	if err != nil {
		return nil, fmt.Errorf("qwen3next: decode config: %w", err)
	}
	a.quant = quant
	if err := a.validate(); err != nil {
		return nil, err
	}
	return a, nil
}

func (a *Qwen3NextArgs) validate() error {
	switch {
	case a.HiddenSize <= 0:
		return fmt.Errorf("qwen3next: hidden_size must be positive, got %d", a.HiddenSize)
	case a.VocabSize <= 0:
		return fmt.Errorf("qwen3next: vocab_size must be positive, got %d", a.VocabSize)
	case a.NumHiddenLayers <= 0:
		return fmt.Errorf("qwen3next: num_hidden_layers must be positive, got %d", a.NumHiddenLayers)
	case a.NumAttentionHeads <= 0:
		return fmt.Errorf("qwen3next: num_attention_heads must be positive, got %d", a.NumAttentionHeads)
	case a.HeadDim <= 0:
		return fmt.Errorf("qwen3next: head_dim must be positive, got %d", a.HeadDim)
	case a.FullAttentionInterval <= 0:
		return fmt.Errorf("qwen3next: full_attention_interval must be positive, got %d", a.FullAttentionInterval)
	case a.LinearNumKeyHeads <= 0 || a.LinearNumValueHeads <= 0:
		return fmt.Errorf("qwen3next: linear head counts must be positive")
	case a.LinearNumValueHeads%a.LinearNumKeyHeads != 0:
		return fmt.Errorf("qwen3next: linear_num_value_heads (%d) must be a multiple of linear_num_key_heads (%d)",
			a.LinearNumValueHeads, a.LinearNumKeyHeads)
	case a.LinearKeyHeadDim <= 0 || a.LinearValueHeadDim <= 0:
		return fmt.Errorf("qwen3next: linear head dims must be positive")
	case a.LinearConvKernelDim <= 0:
		return fmt.Errorf("qwen3next: linear_conv_kernel_dim must be positive, got %d", a.LinearConvKernelDim)
	case a.NumExperts > 0 && a.DecoderSparseStep <= 0:
		return fmt.Errorf("qwen3next: decoder_sparse_step must be positive, got %d", a.DecoderSparseStep)
	case a.NumExperts > 0 && a.NumExpertsPerTok > a.NumExperts:
		return fmt.Errorf("qwen3next: num_experts_per_tok (%d) exceeds num_experts (%d)",
			a.NumExpertsPerTok, a.NumExperts)
	}
	return nil
}

// NumLayers is the decoder depth.
func (a *Qwen3NextArgs) NumLayers() int { return a.NumHiddenLayers }

// IsLinear reports whether layer idx is a gated delta net (the recurrent mixer).
// Every full_attention_interval-th layer is full attention; the rest are linear.
func (a *Qwen3NextArgs) IsLinear(idx int) bool {
	return (idx+1)%a.FullAttentionInterval != 0
}

// IsMoELayer reports whether layer idx uses the sparse mixture: the model must
// have experts, the layer must not be forced dense by mlp_only_layers, and it
// must fall on the decoder_sparse_step cadence.
func (a *Qwen3NextArgs) IsMoELayer(idx int) bool {
	if a.NumExperts <= 0 || slices.Contains(a.MLPOnlyLayers, idx) {
		return false
	}
	return (idx+1)%a.DecoderSparseStep == 0
}

// KeyDim is the gated delta net key width (linear_key_head_dim * linear_num_key_heads).
func (a *Qwen3NextArgs) KeyDim() int { return a.LinearKeyHeadDim * a.LinearNumKeyHeads }

// ValueDim is the gated delta net value width (linear_value_head_dim * linear_num_value_heads).
func (a *Qwen3NextArgs) ValueDim() int { return a.LinearValueHeadDim * a.LinearNumValueHeads }

// ConvDim is the depthwise convolution channel count: two key bands plus one
// value band (the q, k, and v streams the conv runs over).
func (a *Qwen3NextArgs) ConvDim() int { return a.KeyDim()*2 + a.ValueDim() }

// InProjQKVZOut is the fused in_proj_qkvz output width: q and k each a key band,
// v and the output gate z each a value band.
func (a *Qwen3NextArgs) InProjQKVZOut() int { return a.KeyDim()*2 + a.ValueDim()*2 }

// InProjBAOut is the fused in_proj_ba output width: a beta and an alpha scalar per
// value head.
func (a *Qwen3NextArgs) InProjBAOut() int { return a.LinearNumValueHeads * 2 }

// QKVZSplits returns the split boundaries fix_query_key_value_ordering uses on the
// per-key-head qkvz block: q ends at head_k_dim, k at 2*head_k_dim, v at
// 2*head_k_dim + (num_v_heads/num_k_heads)*head_v_dim, and z takes the remainder.
func (a *Qwen3NextArgs) QKVZSplits() []int {
	dn := a.LinearKeyHeadDim
	vPerK := a.LinearNumValueHeads / a.LinearNumKeyHeads
	return []int{dn, 2 * dn, 2*dn + vPerK*a.LinearValueHeadDim}
}

// BASplits returns the single split boundary in_proj_ba uses per key head, between
// the beta and alpha bands (num_v_heads/num_k_heads each).
func (a *Qwen3NextArgs) BASplits() []int {
	return []int{a.LinearNumValueHeads / a.LinearNumKeyHeads}
}

// QProjOut is the attention query projection output: two head bands per attention
// head, one for the query and one for the output gate.
func (a *Qwen3NextArgs) QProjOut() int { return a.NumAttentionHeads * a.HeadDim * 2 }

// RopeDims is the rotated width of the attention rotary embedding.
func (a *Qwen3NextArgs) RopeDims() int { return int(float64(a.HeadDim) * a.PartialRotaryFactor) }

// AttentionScale is head_dim raised to the -1/2 power.
func (a *Qwen3NextArgs) AttentionScale() float64 {
	return 1.0 / math.Sqrt(float64(a.HeadDim))
}

// MakeCache builds the per-layer cache list: a non-trimmable ArraysCache (two
// slots, the conv window and the recurrent state) for each linear layer and a
// growing KVCache for each attention layer. Because at least one entry is not
// trimmable, the list as a whole cannot be trimmed, so the family relies on
// boundary-snapshot prefix caching rather than plain trimming.
func (a *Qwen3NextArgs) MakeCache() []compute.Cache {
	caches := make([]compute.Cache, a.NumLayers())
	for i := range caches {
		if a.IsLinear(i) {
			caches[i] = compute.NewArraysCache(2)
		} else {
			caches[i] = &compute.KVCache{}
		}
	}
	return caches
}

// WeightNames returns the sorted parameter key set after the pre-load patch has
// run. Each block has the two layernorms and an MLP (a dense SwiGLU, or the gate,
// the stacked switch_mlp, and the shared expert with its gate). A linear layer
// adds the gated delta net (the qkvz and ba input projections, the depthwise
// conv, the per-head dt_bias and A_log, the gated norm, and the output
// projection); an attention layer adds the gated query projection, the key and
// value projections, the query and key norms, and the output projection, with the
// attention biases when enabled. Then the embedding, the final norm, and the
// untied head.
func (a *Qwen3NextArgs) WeightNames() []string {
	names := []string{"model.embed_tokens.weight", "model.norm.weight"}
	if !a.TieWordEmbeddings {
		names = append(names, "lm_head.weight")
	}
	for i := range a.NumLayers() {
		p := fmt.Sprintf("model.layers.%d.", i)
		names = append(names, p+"input_layernorm.weight", p+"post_attention_layernorm.weight")
		if a.IsLinear(i) {
			names = append(names, a.linearAttnNames(p+"linear_attn.")...)
		} else {
			names = append(names, a.attnNames(p+"self_attn.")...)
		}
		names = append(names, a.mlpNames(p+"mlp.", i)...)
	}
	sort.Strings(names)
	return names
}

func (a *Qwen3NextArgs) linearAttnNames(p string) []string {
	return []string{
		p + "A_log",
		p + "conv1d.weight",
		p + "dt_bias",
		p + "in_proj_ba.weight",
		p + "in_proj_qkvz.weight",
		p + "norm.weight",
		p + "out_proj.weight",
	}
}

func (a *Qwen3NextArgs) attnNames(p string) []string {
	names := []string{
		p + "q_proj.weight",
		p + "k_proj.weight",
		p + "v_proj.weight",
		p + "o_proj.weight",
		p + "q_norm.weight",
		p + "k_norm.weight",
	}
	if a.AttentionBias {
		names = append(names, p+"q_proj.bias", p+"k_proj.bias", p+"v_proj.bias", p+"o_proj.bias")
	}
	return names
}

func (a *Qwen3NextArgs) mlpNames(p string, idx int) []string {
	if !a.IsMoELayer(idx) {
		return []string{p + "gate_proj.weight", p + "up_proj.weight", p + "down_proj.weight"}
	}
	return []string{
		p + "gate.weight",
		p + "shared_expert.gate_proj.weight",
		p + "shared_expert.up_proj.weight",
		p + "shared_expert.down_proj.weight",
		p + "shared_expert_gate.weight",
		p + "switch_mlp.gate_proj.weight",
		p + "switch_mlp.up_proj.weight",
		p + "switch_mlp.down_proj.weight",
	}
}

// Sanitize is the seam for the pre-load patch. A raw checkpoint carries per-expert
// MLP tensors (mlp.experts.{e}.{gate,up,down}_proj) that the reference stacks into
// a single switch_mlp, a depthwise conv weight that may need its kernel axis moved,
// the multi-token-prediction (mtp) tensors that are dropped, and fused RMSNorm
// weights that are re-centered by adding one. Those are tensor operations that
// need the backend, so they land with the numeric forward; an already-patched
// checkpoint (switch_mlp present) passes through unchanged, which is what the
// reference does too.
func (a *Qwen3NextArgs) Sanitize(weights map[string]*mlxgo.Array) map[string]*mlxgo.Array {
	return weights
}
