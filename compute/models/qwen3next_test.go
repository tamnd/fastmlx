// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/json"
	"maps"
	"os"
	"reflect"
	"testing"

	"github.com/tamnd/fastmlx/compute"
)

// The fixture is captured from the reference qwen3_next model: args from
// ModelArgs.from_dict, the per-layer linear-vs-attention typing and dense-vs-moe
// MLP typing read off the built model, the gated delta net geometry (the key and
// value widths, the conv channels, the fused input projection widths, and the
// fix_query_key_value_ordering split boundaries), the weight names from the
// flattened parameter tree, and the per-layer cache kinds from make_cache.
type qwen3NextFixture struct {
	Derivations []struct {
		Label             string          `json:"label"`
		Config            json.RawMessage `json:"config"`
		IsLinear          []bool          `json:"is_linear"`
		IsMoE             []bool          `json:"is_moe"`
		KeyDim            int             `json:"key_dim"`
		ValueDim          int             `json:"value_dim"`
		ConvDim           int             `json:"conv_dim"`
		InProjQKVZOut     int             `json:"in_proj_qkvz_out"`
		InProjBAOut       int             `json:"in_proj_ba_out"`
		QKVZSplits        []int           `json:"qkvz_splits"`
		BASplits          []int           `json:"ba_splits"`
		WeightNames       []string        `json:"weight_names"`
		CacheCount        int             `json:"cache_count"`
		CacheTypes        []string        `json:"cache_types"`
		TieWordEmbeddings bool            `json:"tie_word_embeddings"`
	} `json:"derivations"`
}

func loadQwen3NextFixture(t *testing.T) qwen3NextFixture {
	t.Helper()
	b, err := os.ReadFile("../testdata/qwen3next_args.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f qwen3NextFixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return f
}

func TestQwen3NextLayerTypingParity(t *testing.T) {
	f := loadQwen3NextFixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParseQwen3NextArgs(d.Config)
			if err != nil {
				t.Fatalf("ParseQwen3NextArgs: %v", err)
			}
			for i := range a.NumLayers() {
				if a.IsLinear(i) != d.IsLinear[i] {
					t.Errorf("IsLinear(%d) = %v, want %v", i, a.IsLinear(i), d.IsLinear[i])
				}
				if a.IsMoELayer(i) != d.IsMoE[i] {
					t.Errorf("IsMoELayer(%d) = %v, want %v", i, a.IsMoELayer(i), d.IsMoE[i])
				}
			}
		})
	}
}

func TestQwen3NextGeometryParity(t *testing.T) {
	f := loadQwen3NextFixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParseQwen3NextArgs(d.Config)
			if err != nil {
				t.Fatalf("ParseQwen3NextArgs: %v", err)
			}
			checkInt(t, "key_dim", a.KeyDim(), d.KeyDim)
			checkInt(t, "value_dim", a.ValueDim(), d.ValueDim)
			checkInt(t, "conv_dim", a.ConvDim(), d.ConvDim)
			checkInt(t, "in_proj_qkvz_out", a.InProjQKVZOut(), d.InProjQKVZOut)
			checkInt(t, "in_proj_ba_out", a.InProjBAOut(), d.InProjBAOut)
			if !reflect.DeepEqual(a.QKVZSplits(), d.QKVZSplits) {
				t.Errorf("qkvz_splits = %v, want %v", a.QKVZSplits(), d.QKVZSplits)
			}
			if !reflect.DeepEqual(a.BASplits(), d.BASplits) {
				t.Errorf("ba_splits = %v, want %v", a.BASplits(), d.BASplits)
			}
		})
	}
}

func TestQwen3NextWeightNamesParity(t *testing.T) {
	f := loadQwen3NextFixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParseQwen3NextArgs(d.Config)
			if err != nil {
				t.Fatalf("ParseQwen3NextArgs: %v", err)
			}
			got := a.WeightNames()
			if !reflect.DeepEqual(got, d.WeightNames) {
				t.Errorf("weight names mismatch (%d vs %d)\n got %v\nwant %v",
					len(got), len(d.WeightNames), got, d.WeightNames)
			}
		})
	}
}

func TestQwen3NextMakeCacheParity(t *testing.T) {
	f := loadQwen3NextFixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParseQwen3NextArgs(d.Config)
			if err != nil {
				t.Fatalf("ParseQwen3NextArgs: %v", err)
			}
			caches := a.MakeCache()
			if len(caches) != d.CacheCount {
				t.Fatalf("cache count = %d, want %d", len(caches), d.CacheCount)
			}
			for i, c := range caches {
				got := cacheKindName(c)
				if got != d.CacheTypes[i] {
					t.Errorf("cache[%d] = %s, want %s", i, got, d.CacheTypes[i])
				}
				// The recurrent layers must carry a non-trimmable cache, the attention
				// layers a trimmable one.
				if a.IsLinear(i) && c.IsTrimmable() {
					t.Errorf("linear layer %d cache should not be trimmable", i)
				}
				if !a.IsLinear(i) && !c.IsTrimmable() {
					t.Errorf("attention layer %d cache should be trimmable", i)
				}
			}
		})
	}
}

func cacheKindName(c compute.Cache) string {
	switch c.(type) {
	case *compute.ArraysCache:
		return "ArraysCache"
	case *compute.KVCache:
		return "KVCache"
	case *compute.RotatingKVCache:
		return "RotatingKVCache"
	default:
		return "unknown"
	}
}

func TestQwen3NextCacheListNotTrimmable(t *testing.T) {
	// A hybrid stack mixes a non-trimmable recurrent cache with trimmable
	// attention caches, so the list as a whole must not be trimmable.
	a, err := ParseQwen3NextArgs(baseQwen3NextConfig(map[string]any{}))
	if err != nil {
		t.Fatalf("ParseQwen3NextArgs: %v", err)
	}
	caches := a.MakeCache()
	allTrimmable := true
	for _, c := range caches {
		if !c.IsTrimmable() {
			allTrimmable = false
		}
	}
	if allTrimmable {
		t.Error("a hybrid cache list should contain a non-trimmable entry")
	}
}

func TestArraysCacheBookkeeping(t *testing.T) {
	c := compute.NewArraysCache(2)
	if c.Slots != 2 {
		t.Errorf("slots = %d, want 2", c.Slots)
	}
	if !c.Empty() {
		t.Error("a fresh recurrent cache should be empty")
	}
	if c.IsTrimmable() {
		t.Error("a recurrent cache is never trimmable")
	}
	if c.Capacity() != 0 || c.Size() != 0 {
		t.Errorf("capacity/size = %d/%d, want 0/0", c.Capacity(), c.Size())
	}
	plan := c.Update(3)
	if plan.Prev != 0 || plan.Offset != 3 {
		t.Errorf("update plan prev/offset = %d/%d, want 0/3", plan.Prev, plan.Offset)
	}
	if c.Empty() {
		t.Error("the cache should not be empty after a step")
	}
	if c.Trim(2) != 0 {
		t.Error("trimming a recurrent cache must drop nothing")
	}
	if c.Size() != 0 {
		t.Error("size stays 0 (the recurrent state is not a sequence)")
	}
}

func TestParseQwen3NextDefaults(t *testing.T) {
	// full_attention_interval, norm_topk_prob, tie, attention_bias, rope_theta, and
	// partial_rotary_factor are all absent.
	cfg := `{"model_type":"qwen3_next","hidden_size":64,"num_hidden_layers":4,` +
		`"intermediate_size":128,"num_attention_heads":8,"head_dim":16,"vocab_size":320,` +
		`"linear_num_value_heads":8,"linear_num_key_heads":4,"linear_key_head_dim":16,` +
		`"linear_value_head_dim":16,"linear_conv_kernel_dim":4,"num_experts":0,` +
		`"num_experts_per_tok":2,"decoder_sparse_step":1,"shared_expert_intermediate_size":64,` +
		`"moe_intermediate_size":32,"num_key_value_heads":4,"max_position_embeddings":4096}`
	a, err := ParseQwen3NextArgs([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseQwen3NextArgs: %v", err)
	}
	checkInt(t, "full_attention_interval", a.FullAttentionInterval, 4)
	checkFloat(t, "rope_theta", a.RopeTheta, 10000)
	checkFloat(t, "partial_rotary_factor", a.PartialRotaryFactor, 1.0)
	if a.NormTopkProb || a.TieWordEmbeddings || a.AttentionBias {
		t.Error("norm_topk_prob, tie, attention_bias should default false")
	}
	// No experts: every layer is a dense MLP.
	for i := range a.NumLayers() {
		if a.IsMoELayer(i) {
			t.Errorf("dense model should have no moe layer, got moe at %d", i)
		}
	}
}

func TestParseQwen3NextErrors(t *testing.T) {
	cases := []struct {
		name      string
		overrides map[string]any
		raw       string
	}{
		{name: "bad_json", raw: `{`},
		{name: "no_hidden", overrides: map[string]any{"hidden_size": 0}},
		{name: "no_layers", overrides: map[string]any{"num_hidden_layers": 0}},
		{name: "bad_interval", overrides: map[string]any{"full_attention_interval": 0}},
		{name: "v_not_multiple_of_k", overrides: map[string]any{"linear_num_value_heads": 6, "linear_num_key_heads": 4}},
		{name: "bad_conv_kernel", overrides: map[string]any{"linear_conv_kernel_dim": 0}},
		{name: "too_many_experts_per_tok", overrides: map[string]any{"num_experts": 4, "num_experts_per_tok": 9}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := []byte(c.raw)
			if cfg == nil {
				cfg = baseQwen3NextConfig(c.overrides)
			}
			if _, err := ParseQwen3NextArgs(cfg); err == nil {
				t.Error("expected error")
			}
		})
	}
}

func TestQwen3NextSanitizeIdentity(t *testing.T) {
	a, err := ParseQwen3NextArgs(baseQwen3NextConfig(map[string]any{}))
	if err != nil {
		t.Fatalf("ParseQwen3NextArgs: %v", err)
	}
	if got := a.Sanitize(nil); got != nil {
		t.Error("Sanitize should pass through unchanged at this stage")
	}
}

func BenchmarkParseQwen3NextArgs(b *testing.B) {
	b.ReportAllocs()
	cfg := baseQwen3NextConfig(map[string]any{
		"hidden_size":         2048,
		"num_hidden_layers":   48,
		"num_experts":         512,
		"num_experts_per_tok": 10,
	})
	for b.Loop() {
		a, err := ParseQwen3NextArgs(cfg)
		if err != nil {
			b.Fatal(err)
		}
		_ = a.WeightNames()
	}
}

// baseQwen3NextConfig builds a small valid hybrid config and applies overrides.
func baseQwen3NextConfig(overrides map[string]any) []byte {
	cfg := map[string]any{
		"model_type":                      "qwen3_next",
		"hidden_size":                     64,
		"num_hidden_layers":               4,
		"intermediate_size":               128,
		"num_attention_heads":             8,
		"linear_num_value_heads":          8,
		"linear_num_key_heads":            4,
		"linear_key_head_dim":             16,
		"linear_value_head_dim":           16,
		"linear_conv_kernel_dim":          4,
		"num_experts":                     4,
		"num_experts_per_tok":             2,
		"decoder_sparse_step":             1,
		"shared_expert_intermediate_size": 64,
		"mlp_only_layers":                 []int{},
		"moe_intermediate_size":           32,
		"rms_norm_eps":                    1e-6,
		"vocab_size":                      320,
		"num_key_value_heads":             4,
		"rope_theta":                      10000.0,
		"partial_rotary_factor":           0.5,
		"max_position_embeddings":         4096,
		"head_dim":                        16,
		"norm_topk_prob":                  true,
		"full_attention_interval":         4,
	}
	maps.Copy(cfg, overrides)
	b, err := json.Marshal(cfg)
	if err != nil {
		panic(err)
	}
	return b
}
