// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"reflect"
	"sort"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// The fixture is captured from the reference deepseek_v3 model: args from
// ModelArgs.from_dict, the multi-head latent attention geometry (the per-head
// query width, whether the low-rank query path is active, the kv_a projection
// width), the attention scale read straight off the built module so the yarn
// mscale adjustment is included, the per-layer dense-or-moe typing, the weight
// names from the flattened parameter tree, the cache kinds from
// make_prompt_cache, and a set of group_expert_select routing cases.
type deepseekFixture struct {
	Derivations []struct {
		Label       string          `json:"label"`
		Config      json.RawMessage `json:"config"`
		QHeadDim    int             `json:"q_head_dim"`
		HasQLora    bool            `json:"has_q_lora"`
		KVAOut      int             `json:"kv_a_out"`
		Scale       float64         `json:"scale"`
		LayerTypes  []string        `json:"layer_types"`
		WeightNames []string        `json:"weight_names"`
		CacheCount  int             `json:"cache_count"`
		CacheTypes  []string        `json:"cache_types"`
	} `json:"derivations"`
	Routing []struct {
		Label               string    `json:"label"`
		Gates               []float32 `json:"gates"`
		Bias                []float32 `json:"bias"`
		TopK                int       `json:"top_k"`
		NGroup              int       `json:"n_group"`
		TopkGroup           int       `json:"topk_group"`
		RoutedScalingFactor float64   `json:"routed_scaling_factor"`
		NormTopkProb        bool      `json:"norm_topk_prob"`
		Inds                []int     `json:"inds"`
		Scores              []float64 `json:"scores"`
	} `json:"routing"`
}

func loadDeepseekFixture(t *testing.T) deepseekFixture {
	t.Helper()
	b, err := os.ReadFile("../testdata/deepseek_args.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f deepseekFixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return f
}

func TestDeepseekV3ArgsParity(t *testing.T) {
	f := loadDeepseekFixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParseDeepseekV3Args(d.Config)
			if err != nil {
				t.Fatalf("ParseDeepseekV3Args: %v", err)
			}
			checkInt(t, "q_head_dim", a.QHeadDim(), d.QHeadDim)
			checkInt(t, "kv_a_out", a.KVAOut(), d.KVAOut)
			if a.HasQLora() != d.HasQLora {
				t.Errorf("has_q_lora = %v, want %v", a.HasQLora(), d.HasQLora)
			}
			// The scale carries the yarn mscale adjustment; the reference computes it
			// in float64, so it matches at the default epsilon.
			checkFloat(t, "scale", a.AttentionScale(), d.Scale)
			if !reflect.DeepEqual(a.LayerTypes(), d.LayerTypes) {
				t.Errorf("layer types = %v, want %v", a.LayerTypes(), d.LayerTypes)
			}
		})
	}
}

func TestDeepseekV3WeightNamesParity(t *testing.T) {
	f := loadDeepseekFixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParseDeepseekV3Args(d.Config)
			if err != nil {
				t.Fatalf("ParseDeepseekV3Args: %v", err)
			}
			got := a.WeightNames()
			want := append([]string(nil), d.WeightNames...)
			// The reference parameter tree is already sorted; sort the captured set
			// defensively so the comparison is order-independent.
			if !reflect.DeepEqual(got, sortedCopy(want)) {
				t.Errorf("weight names mismatch (%d vs %d)\n got %v\nwant %v",
					len(got), len(want), got, want)
			}
		})
	}
}

func TestDeepseekV3MakeCacheParity(t *testing.T) {
	f := loadDeepseekFixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParseDeepseekV3Args(d.Config)
			if err != nil {
				t.Fatalf("ParseDeepseekV3Args: %v", err)
			}
			caches := a.MakeCache()
			if len(caches) != d.CacheCount {
				t.Fatalf("cache count = %d, want %d", len(caches), d.CacheCount)
			}
			for i := range caches {
				if d.CacheTypes[i] != "KVCache" {
					t.Errorf("cache[%d] expected KVCache, fixture says %s", i, d.CacheTypes[i])
				}
			}
		})
	}
}

func TestDeepseekV3GroupExpertSelectParity(t *testing.T) {
	f := loadDeepseekFixture(t)
	for _, r := range f.Routing {
		t.Run(r.Label, func(t *testing.T) {
			inds, scores := GroupExpertSelect(r.Gates, r.Bias, r.TopK, r.NGroup,
				r.TopkGroup, r.RoutedScalingFactor, r.NormTopkProb)
			if len(inds) != len(r.Inds) {
				t.Fatalf("selected %d experts, want %d", len(inds), len(r.Inds))
			}
			// Order within the top-k is not significant (the caller forms a weighted
			// sum), so compare the expert-to-weight mapping rather than the order.
			got := map[int]float64{}
			for i, ix := range inds {
				got[ix] = scores[i]
			}
			want := map[int]float64{}
			for i, ix := range r.Inds {
				want[ix] = r.Scores[i]
			}
			for ix, w := range want {
				g, ok := got[ix]
				if !ok {
					t.Errorf("expert %d missing from selection %v", ix, inds)
					continue
				}
				// Scores follow a float32 sigmoid in the reference; compare at a
				// float32-relative tolerance.
				if rel := math.Abs(g-w) / math.Abs(w); rel > 1e-5 {
					t.Errorf("expert %d weight = %v, want %v (rel %g)", ix, g, w, rel)
				}
			}
		})
	}
}

func TestDeepseekV3IsMoELayer(t *testing.T) {
	// first_k_dense_replace=2, moe_layer_freq=2: layers 0,1 dense, then moe only
	// on even indices past the prefix.
	cfg := baseDeepseekConfig(map[string]any{
		"num_hidden_layers":     6,
		"first_k_dense_replace": 2,
		"moe_layer_freq":        2,
	})
	a, err := ParseDeepseekV3Args(cfg)
	if err != nil {
		t.Fatalf("ParseDeepseekV3Args: %v", err)
	}
	want := []bool{false, false, true, false, true, false}
	for i, w := range want {
		if a.IsMoELayer(i) != w {
			t.Errorf("IsMoELayer(%d) = %v, want %v", i, a.IsMoELayer(i), w)
		}
	}
}

func TestParseDeepseekV3QLoraNullVersusAbsent(t *testing.T) {
	// An absent q_lora_rank keeps the low-rank default; an explicit null selects
	// the plain q_proj path.
	absent := baseDeepseekConfig(map[string]any{})
	a, err := ParseDeepseekV3Args(absent)
	if err != nil {
		t.Fatalf("ParseDeepseekV3Args(absent): %v", err)
	}
	if !a.HasQLora() || a.QLoraRank != 24 {
		t.Errorf("absent q_lora_rank: HasQLora=%v rank=%d, want true/24", a.HasQLora(), a.QLoraRank)
	}

	withNull := baseDeepseekConfig(map[string]any{"q_lora_rank": nil})
	b, err := ParseDeepseekV3Args(withNull)
	if err != nil {
		t.Fatalf("ParseDeepseekV3Args(null): %v", err)
	}
	if b.HasQLora() || b.QLoraRank != 0 {
		t.Errorf("null q_lora_rank: HasQLora=%v rank=%d, want false/0", b.HasQLora(), b.QLoraRank)
	}
}

func TestParseDeepseekV3Defaults(t *testing.T) {
	// A minimal dense config: no experts, so it is a plain stack, kv heads track
	// the attention heads, and the routing fields take their dataclass defaults.
	cfg := `{"model_type":"deepseek_v3","hidden_size":64,"num_hidden_layers":2,` +
		`"intermediate_size":128,"num_attention_heads":8,"vocab_size":320,"kv_lora_rank":16}`
	a, err := ParseDeepseekV3Args([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseDeepseekV3Args: %v", err)
	}
	if a.IsMoE() {
		t.Error("no n_routed_experts should mean a dense model")
	}
	checkInt(t, "num_key_value_heads", a.NumKeyValueHeads, 8)
	checkInt(t, "qk_rope_head_dim", a.QKRopeHeadDim, 64)
	checkInt(t, "qk_nope_head_dim", a.QKNopeHeadDim, 128)
	checkFloat(t, "routed_scaling_factor", a.RoutedScalingFactor, 1.0)
	if !a.NormTopkProb {
		t.Error("norm_topk_prob should default to true")
	}
	if a.RopeScaling != nil {
		t.Error("rope_scaling should be nil when absent")
	}
	for i := range a.NumLayers() {
		if a.IsMoELayer(i) {
			t.Errorf("dense model should have no moe layer, got moe at %d", i)
		}
	}
}

func TestParseDeepseekV3Errors(t *testing.T) {
	cases := []struct {
		name string
		cfg  string
	}{
		{"bad_json", `{`},
		{"no_hidden", `{"num_attention_heads":8,"num_hidden_layers":2,"vocab_size":320,"kv_lora_rank":16}`},
		{"no_layers", `{"hidden_size":64,"num_attention_heads":8,"vocab_size":320,"kv_lora_rank":16}`},
		{"bad_topk_method", `{"hidden_size":64,"num_attention_heads":8,"num_hidden_layers":2,"vocab_size":320,"kv_lora_rank":16,"topk_method":"greedy"}`},
		{"experts_not_group_multiple", `{"hidden_size":64,"num_attention_heads":8,"num_hidden_layers":2,"vocab_size":320,"kv_lora_rank":16,"n_routed_experts":7,"n_group":4,"topk_group":2,"num_experts_per_tok":2}`},
		{"topk_group_exceeds", `{"hidden_size":64,"num_attention_heads":8,"num_hidden_layers":2,"vocab_size":320,"kv_lora_rank":16,"n_routed_experts":8,"n_group":2,"topk_group":4,"num_experts_per_tok":2}`},
		{"too_many_experts_per_tok", `{"hidden_size":64,"num_attention_heads":8,"num_hidden_layers":2,"vocab_size":320,"kv_lora_rank":16,"n_routed_experts":8,"n_group":4,"topk_group":2,"num_experts_per_tok":9}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := ParseDeepseekV3Args([]byte(c.cfg)); err == nil {
				t.Error("expected error")
			}
		})
	}
}

// Sanitize drops the multi-token-prediction layer (the index past the decoder
// stack) and any rotary inverse-frequency buffer, and keeps every other key.
func TestDeepseekV3SanitizeDrops(t *testing.T) {
	a, err := ParseDeepseekV3Args(baseDeepseekConfig(map[string]any{"num_hidden_layers": 4}))
	if err != nil {
		t.Fatalf("ParseDeepseekV3Args: %v", err)
	}
	mk := func() *mlxgo.Array {
		arr, err := mlxgo.NewFloat32([]float32{0}, 1)
		if err != nil {
			t.Fatalf("NewFloat32: %v", err)
		}
		return arr
	}
	w := map[string]*mlxgo.Array{
		"model.embed_tokens.weight":                     mk(),
		"model.layers.0.input_layernorm.weight":         mk(),
		"model.layers.0.self_attn.rotary_emb.inv_freq":  mk(), // dropped
		"model.layers.4.input_layernorm.weight":         mk(), // MTP layer, dropped
		"model.layers.4.mlp.experts.0.gate_proj.weight": mk(), // MTP layer, dropped
		"lm_head.weight":                                mk(),
	}
	got := a.Sanitize(w)
	want := []string{
		"model.embed_tokens.weight",
		"model.layers.0.input_layernorm.weight",
		"lm_head.weight",
	}
	if len(got) != len(want) {
		t.Fatalf("Sanitize kept %d keys, want %d (%v)", len(got), len(want), keysOf(got))
	}
	for _, k := range want {
		if _, ok := got[k]; !ok {
			t.Errorf("Sanitize dropped %q, want kept", k)
		}
	}
}

// stackExperts joins the per-expert MLP tensors along a new expert axis. The stub
// reaches the unavailable error at the Stack kernel, a fully-stacked checkpoint
// passes through with no kernel, and a cohort missing an expert is caught host-side
// before any kernel.
func TestStackExperts(t *testing.T) {
	mk := func() *mlxgo.Array {
		arr, err := mlxgo.NewFloat32([]float32{0}, 1)
		if err != nil {
			t.Fatalf("NewFloat32: %v", err)
		}
		return arr
	}

	// One MoE layer with all three projections present for every expert.
	const numExperts = 3
	perExpert := map[string]*mlxgo.Array{}
	for _, proj := range []string{"gate_proj", "down_proj", "up_proj"} {
		for e := 0; e < numExperts; e++ {
			perExpert[fmt.Sprintf("model.layers.0.mlp.experts.%d.%s.weight", e, proj)] = mk()
		}
	}
	if err := stackExperts(perExpert, 1, numExperts, mlxgo.DefaultStream()); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("stackExperts on the stub: err = %v, want ErrMLXUnavailable at the Stack kernel", err)
	}

	// A checkpoint already carrying stacked switch_mlp tensors has no per-expert
	// key, so stackExperts touches no kernel and leaves the map unchanged.
	stacked := map[string]*mlxgo.Array{
		"model.layers.0.mlp.switch_mlp.gate_proj.weight": mk(),
	}
	before := keysOf(stacked)
	if err := stackExperts(stacked, 1, numExperts, mlxgo.DefaultStream()); err != nil {
		t.Fatalf("stackExperts on a stacked checkpoint: %v", err)
	}
	if after := keysOf(stacked); !reflect.DeepEqual(before, after) {
		t.Errorf("stacked checkpoint mutated: %v -> %v", before, after)
	}

	// Expert 0 present but the cohort incomplete: caught before the kernel.
	missing := map[string]*mlxgo.Array{
		"model.layers.0.mlp.experts.0.gate_proj.weight": mk(),
		"model.layers.0.mlp.experts.1.gate_proj.weight": mk(),
	}
	err := stackExperts(missing, 1, numExperts, mlxgo.DefaultStream())
	if err == nil || errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("incomplete cohort: err = %v, want a host-side missing-expert error", err)
	}
}

func keysOf(m map[string]*mlxgo.Array) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func BenchmarkParseDeepseekV3Args(b *testing.B) {
	b.ReportAllocs()
	cfg := baseDeepseekConfig(map[string]any{
		"hidden_size":         7168,
		"num_hidden_layers":   61,
		"num_attention_heads": 128,
		"n_routed_experts":    256,
		"n_group":             8,
		"topk_group":          4,
		"num_experts_per_tok": 8,
		"vocab_size":          129280,
	})
	for b.Loop() {
		a, err := ParseDeepseekV3Args(cfg)
		if err != nil {
			b.Fatal(err)
		}
		_ = a.WeightNames()
	}
}

func BenchmarkGroupExpertSelect(b *testing.B) {
	b.ReportAllocs()
	gates := make([]float32, 256)
	bias := make([]float32, 256)
	for i := range gates {
		gates[i] = float32(i%17) * 0.1
		bias[i] = float32(i%5) * 0.01
	}
	for b.Loop() {
		GroupExpertSelect(gates, bias, 8, 8, 4, 2.5, true)
	}
}

func sortedCopy(in []string) []string {
	out := append([]string(nil), in...)
	sort.Strings(out)
	return out
}

// baseDeepseekConfig builds a small valid config with the moe routing fields set,
// applying the given overrides. Values are merged then marshalled to JSON.
func baseDeepseekConfig(overrides map[string]any) []byte {
	cfg := map[string]any{
		"model_type":              "deepseek_v3",
		"vocab_size":              320,
		"hidden_size":             64,
		"intermediate_size":       128,
		"moe_intermediate_size":   32,
		"num_hidden_layers":       4,
		"num_attention_heads":     8,
		"num_key_value_heads":     8,
		"n_shared_experts":        1,
		"n_routed_experts":        8,
		"routed_scaling_factor":   2.5,
		"kv_lora_rank":            16,
		"q_lora_rank":             24,
		"qk_rope_head_dim":        8,
		"v_head_dim":              16,
		"qk_nope_head_dim":        16,
		"topk_method":             "noaux_tc",
		"scoring_func":            "sigmoid",
		"norm_topk_prob":          true,
		"n_group":                 4,
		"topk_group":              2,
		"num_experts_per_tok":     4,
		"moe_layer_freq":          1,
		"first_k_dense_replace":   1,
		"max_position_embeddings": 4096,
		"rms_norm_eps":            1e-6,
		"rope_theta":              10000.0,
	}
	maps.Copy(cfg, overrides)
	b, err := json.Marshal(cfg)
	if err != nil {
		panic(err)
	}
	return b
}
