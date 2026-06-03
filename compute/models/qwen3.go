// SPDX-License-Identifier: MIT OR Apache-2.0

// Package models holds the per-architecture model definitions the compute
// backend loads and runs. Each model splits into two halves: a host-testable
// surface (the config decode, the derived dimensions, the weight-name layout,
// and the per-layer cache wiring) that compiles and tests anywhere, and the
// numeric Forward pass that drives the mlxgo tensor ops and therefore lives
// behind the mlx build tag.
//
// Qwen3 dense is the first target. The fields and the weight layout mirror the
// reference Qwen3 model: a decoder stack of attention + SwiGLU MLP blocks with
// per-head RMSNorm on the query and key projections, grouped-query attention,
// and RoPE. The vocabulary head is tied to the embedding table unless the
// config opts out.
package models

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"github.com/tamnd/fastmlx/compute"
	"github.com/tamnd/fastmlx/mlxgo"
)

// Qwen3Args decodes the subset of config.json the Qwen3 model needs. Unknown
// keys (torch_dtype, _name_or_path, and the rest of the HuggingFace blob) are
// ignored by encoding/json, matching the reference BaseModelArgs.from_dict
// behavior of dropping fields the dataclass does not declare.
type Qwen3Args struct {
	ModelType             string      `json:"model_type"`
	HiddenSize            int         `json:"hidden_size"`
	NumHiddenLayers       int         `json:"num_hidden_layers"`
	IntermediateSize      int         `json:"intermediate_size"`
	NumAttentionHeads     int         `json:"num_attention_heads"`
	RMSNormEps            float64     `json:"rms_norm_eps"`
	VocabSize             int         `json:"vocab_size"`
	NumKeyValueHeads      int         `json:"num_key_value_heads"`
	MaxPositionEmbeddings int         `json:"max_position_embeddings"`
	RopeTheta             float64     `json:"rope_theta"`
	HeadDim               int         `json:"head_dim"`
	TieWordEmbeddings     bool        `json:"tie_word_embeddings"`
	RopeScaling           RopeScaling `json:"rope_scaling"`
}

// RopeScaling carries the optional RoPE scaling block. It stays a free-form map
// because the reference passes the whole dict through to the rope initializer,
// where different scaling types (linear, yarn, llama3) read different keys.
type RopeScaling map[string]any

// ParseQwen3Args decodes a config.json body into Qwen3Args, fills the head_dim
// default the reference derives when the key is absent, and validates the
// fields the forward pass relies on.
func ParseQwen3Args(configJSON []byte) (*Qwen3Args, error) {
	var a Qwen3Args
	if err := json.Unmarshal(configJSON, &a); err != nil {
		return nil, fmt.Errorf("qwen3: decode config: %w", err)
	}
	if a.NumAttentionHeads <= 0 {
		return nil, fmt.Errorf("qwen3: num_attention_heads must be positive, got %d", a.NumAttentionHeads)
	}
	if a.HeadDim == 0 {
		a.HeadDim = a.HiddenSize / a.NumAttentionHeads
	}
	if a.NumKeyValueHeads == 0 {
		a.NumKeyValueHeads = a.NumAttentionHeads
	}
	if err := a.validate(); err != nil {
		return nil, err
	}
	return &a, nil
}

func (a *Qwen3Args) validate() error {
	switch {
	case a.HiddenSize <= 0:
		return fmt.Errorf("qwen3: hidden_size must be positive, got %d", a.HiddenSize)
	case a.NumHiddenLayers <= 0:
		return fmt.Errorf("qwen3: num_hidden_layers must be positive, got %d", a.NumHiddenLayers)
	case a.HeadDim <= 0:
		return fmt.Errorf("qwen3: head_dim must be positive, got %d", a.HeadDim)
	case a.VocabSize <= 0:
		return fmt.Errorf("qwen3: vocab_size must be positive, got %d", a.VocabSize)
	case a.NumAttentionHeads%a.NumKeyValueHeads != 0:
		return fmt.Errorf("qwen3: num_attention_heads (%d) must be a multiple of num_key_value_heads (%d)",
			a.NumAttentionHeads, a.NumKeyValueHeads)
	}
	return nil
}

// Scale is the attention logit scale, head_dim raised to the -1/2 power.
func (a *Qwen3Args) Scale() float64 { return math.Pow(float64(a.HeadDim), -0.5) }

// QProjOut is the query projection output width: one head_dim slice per
// attention head.
func (a *Qwen3Args) QProjOut() int { return a.NumAttentionHeads * a.HeadDim }

// KVProjOut is the key (and value) projection output width: one head_dim slice
// per key/value head.
func (a *Qwen3Args) KVProjOut() int { return a.NumKeyValueHeads * a.HeadDim }

// GQARepeat is the grouped-query repeat factor, how many query heads share each
// key/value head.
func (a *Qwen3Args) GQARepeat() int { return a.NumAttentionHeads / a.NumKeyValueHeads }

// MakeCache builds one KV cache per decoder layer. Qwen3 dense uses the plain
// growing cache (no sliding window), so each entry is a zero-value KVCache
// ready to take its first Update.
func (a *Qwen3Args) MakeCache() []*compute.KVCache {
	caches := make([]*compute.KVCache, a.NumHiddenLayers)
	for i := range caches {
		caches[i] = &compute.KVCache{}
	}
	return caches
}

// WeightNames returns the full set of parameter keys the model expects, sorted,
// matching the reference parameter layout: every tensor under the "model."
// prefix plus the untied "lm_head.weight". When the embeddings are tied the
// head reuses the embedding table, so lm_head.weight is absent.
func (a *Qwen3Args) WeightNames() []string {
	names := []string{"model.embed_tokens.weight"}
	for i := range a.NumHiddenLayers {
		p := fmt.Sprintf("model.layers.%d.", i)
		names = append(names,
			p+"input_layernorm.weight",
			p+"mlp.down_proj.weight",
			p+"mlp.gate_proj.weight",
			p+"mlp.up_proj.weight",
			p+"post_attention_layernorm.weight",
			p+"self_attn.k_norm.weight",
			p+"self_attn.k_proj.weight",
			p+"self_attn.o_proj.weight",
			p+"self_attn.q_norm.weight",
			p+"self_attn.q_proj.weight",
			p+"self_attn.v_proj.weight",
		)
	}
	names = append(names, "model.norm.weight")
	if !a.TieWordEmbeddings {
		names = append(names, "lm_head.weight")
	}
	sort.Strings(names)
	return names
}

// Sanitize drops the weights the loader must not feed to the model. Mirroring
// the reference, a tied checkpoint that still ships an explicit lm_head.weight
// has that key removed so the head falls back to the embedding table.
func (a *Qwen3Args) Sanitize(weights map[string]*mlxgo.Array) map[string]*mlxgo.Array {
	if a.TieWordEmbeddings {
		delete(weights, "lm_head.weight")
	}
	return weights
}

// LoadQwen3 assembles a runnable model from a checkpoint: it decodes the
// config, loads the safetensors weights, drops the tied head, and wires the
// result. configJSON is the config.json body; blob is a safetensors container
// (one file, or a shard merged with MergeTensors). The numeric Forward needs
// the mlx backend, but the assembly here runs on any host.
func LoadQwen3(configJSON, blob []byte) (*Qwen3Model, error) {
	args, err := ParseQwen3Args(configJSON)
	if err != nil {
		return nil, err
	}
	weights, err := compute.LoadTensors(blob)
	if err != nil {
		return nil, err
	}
	return NewQwen3Model(args, args.Sanitize(weights))
}
