// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// The fixture is captured from the reference Gemma 4 text model: the per-layer
// derivations from the built Model (layer_types, previous_kvs, has_kv, per-layer
// head dim and kv heads, the K-eq-V flags, and the per-kind rotary theta), the
// cache kinds from make_cache, the weight names from the flattened sanitized
// parameter tree, and the soft-cap values from logit_softcap.
type gemma4Fixture struct {
	Derivations []struct {
		Label                 string          `json:"label"`
		Config                json.RawMessage `json:"config"`
		LayerTypes            []string        `json:"layer_types"`
		NumLayers             int             `json:"num_layers"`
		NumKVSharedLayers     int             `json:"num_kv_shared_layers"`
		FirstKVShared         int             `json:"first_kv_shared"`
		PreviousKVs           []int           `json:"previous_kvs"`
		HasKV                 []bool          `json:"has_kv"`
		PerLayerHeadDim       []int           `json:"per_layer_head_dim"`
		PerLayerNumKVHeads    []int           `json:"per_layer_n_kv_heads"`
		PerLayerIsSliding     []bool          `json:"per_layer_is_sliding"`
		PerLayerUseKEqV       []bool          `json:"per_layer_use_k_eq_v"`
		PerLayerRopeTheta     []float64       `json:"per_layer_rope_theta"`
		PerLayerPartialRotary []float64       `json:"per_layer_partial_rotary"`
		EmbedScale            float64         `json:"embed_scale"`
		TieWordEmbeddings     bool            `json:"tie_word_embeddings"`
		FinalLogitSoftcapping float64         `json:"final_logit_softcapping"`
		CacheCount            int             `json:"cache_count"`
		CacheTypes            []string        `json:"cache_types"`
		WeightNames           []string        `json:"weight_names"`
	} `json:"derivations"`
	Softcap []struct {
		Cap    float64   `json:"cap"`
		X      []float64 `json:"x"`
		Values []float64 `json:"values"`
	} `json:"softcap"`
}

func loadGemma4Fixture(t *testing.T) gemma4Fixture {
	t.Helper()
	b, err := os.ReadFile("../testdata/gemma4_text_args.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f gemma4Fixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return f
}

func TestGemma4DerivationsParity(t *testing.T) {
	f := loadGemma4Fixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParseGemma4TextArgs(d.Config)
			if err != nil {
				t.Fatalf("ParseGemma4TextArgs: %v", err)
			}
			checkInt(t, "num_layers", a.NumLayers(), d.NumLayers)
			checkInt(t, "first_kv_shared", a.FirstKVShared(), d.FirstKVShared)
			if !reflect.DeepEqual(a.LayerTypes, d.LayerTypes) {
				t.Errorf("layer_types = %v, want %v", a.LayerTypes, d.LayerTypes)
			}
			if !reflect.DeepEqual(a.PreviousKVs(), d.PreviousKVs) {
				t.Errorf("previous_kvs = %v, want %v", a.PreviousKVs(), d.PreviousKVs)
			}
			if a.TieWordEmbeddings != d.TieWordEmbeddings {
				t.Errorf("tie_word_embeddings = %v, want %v", a.TieWordEmbeddings, d.TieWordEmbeddings)
			}
			checkFloat(t, "embed_scale", a.EmbedScale(), d.EmbedScale)
			checkFloat(t, "final_logit_softcapping", a.FinalLogitSoftcapping, d.FinalLogitSoftcapping)
			for i := range a.NumLayers() {
				if a.HasKV(i) != d.HasKV[i] {
					t.Errorf("has_kv[%d] = %v, want %v", i, a.HasKV(i), d.HasKV[i])
				}
				if a.IsSliding(i) != d.PerLayerIsSliding[i] {
					t.Errorf("is_sliding[%d] = %v, want %v", i, a.IsSliding(i), d.PerLayerIsSliding[i])
				}
				if a.UseKEqV(i) != d.PerLayerUseKEqV[i] {
					t.Errorf("use_k_eq_v[%d] = %v, want %v", i, a.UseKEqV(i), d.PerLayerUseKEqV[i])
				}
				checkInt(t, "head_dim", a.PerLayerHeadDim(i), d.PerLayerHeadDim[i])
				checkInt(t, "n_kv_heads", a.PerLayerNumKVHeads(i), d.PerLayerNumKVHeads[i])
				checkFloat(t, "rope_theta", a.LayerRopeTheta(i), d.PerLayerRopeTheta[i])
				checkFloat(t, "partial_rotary", a.LayerPartialRotary(i), d.PerLayerPartialRotary[i])
			}
		})
	}
}

func TestGemma4WeightNamesParity(t *testing.T) {
	f := loadGemma4Fixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParseGemma4TextArgs(d.Config)
			if err != nil {
				t.Fatalf("ParseGemma4TextArgs: %v", err)
			}
			got := a.WeightNames()
			if !reflect.DeepEqual(got, d.WeightNames) {
				t.Errorf("weight names mismatch (%d vs %d)\n got %v\nwant %v",
					len(got), len(d.WeightNames), got, d.WeightNames)
			}
		})
	}
}

func TestGemma4MakeCacheParity(t *testing.T) {
	f := loadGemma4Fixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParseGemma4TextArgs(d.Config)
			if err != nil {
				t.Fatalf("ParseGemma4TextArgs: %v", err)
			}
			caches := a.MakeCache()
			if len(caches) != d.CacheCount {
				t.Fatalf("cache count = %d, want %d", len(caches), d.CacheCount)
			}
			for i, kv := range caches {
				if got := cacheTypeName(kv); got != d.CacheTypes[i] {
					t.Errorf("cache[%d] type = %s, want %s", i, got, d.CacheTypes[i])
				}
			}
		})
	}
}

func TestGemma4LogitSoftcapParity(t *testing.T) {
	f := loadGemma4Fixture(t)
	for _, sc := range f.Softcap {
		a := &Gemma4TextArgs{FinalLogitSoftcapping: sc.Cap}
		for i, x := range sc.X {
			got := a.LogitSoftcap(x)
			if math.Abs(got-sc.Values[i]) > 1e-4 {
				t.Errorf("softcap(cap=%v, x=%v) = %v, want %v", sc.Cap, x, got, sc.Values[i])
			}
		}
	}
	// A non-positive cap is a pass-through.
	a := &Gemma4TextArgs{FinalLogitSoftcapping: 0}
	if got := a.LogitSoftcap(123.4); got != 123.4 {
		t.Errorf("disabled softcap = %v, want pass-through", got)
	}
}

func TestGemma4DefaultLayerTypes(t *testing.T) {
	// Pattern 5 over 7 layers: four sliding then one full, tiled and truncated.
	got := gemma4DefaultLayerTypes(5, 7)
	want := []string{
		slidingAttention, slidingAttention, slidingAttention, slidingAttention,
		fullAttention, slidingAttention, slidingAttention,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("default layer types = %v, want %v", got, want)
	}
	// Pattern 1 is all full attention.
	for i, ty := range gemma4DefaultLayerTypes(1, 3) {
		if ty != fullAttention {
			t.Errorf("pattern-1 layer %d = %s, want full", i, ty)
		}
	}
}

func TestParseGemma4Defaults(t *testing.T) {
	// head_dim, global_head_dim, num_key_value_heads, hidden_size_per_layer_input,
	// final_logit_softcapping, tie_word_embeddings, and rope_parameters all absent.
	cfg := `{"model_type":"gemma4_text","hidden_size":256,"num_attention_heads":8,` +
		`"num_hidden_layers":5,"vocab_size":100,"sliding_window":512,"sliding_window_pattern":5}`
	a, err := ParseGemma4TextArgs([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseGemma4TextArgs: %v", err)
	}
	checkInt(t, "head_dim", a.HeadDim, 256)
	checkInt(t, "global_head_dim", a.GlobalHeadDim, 512)
	checkInt(t, "num_key_value_heads", a.NumKeyValueHeads, 1)
	checkInt(t, "hidden_size_per_layer_input", a.HiddenSizePerLayerIn, 256)
	checkInt(t, "vocab_size_per_layer_input", a.VocabSizePerLayerIn, 100)
	checkFloat(t, "final_logit_softcapping", a.FinalLogitSoftcapping, 30.0)
	if !a.TieWordEmbeddings {
		t.Error("tie_word_embeddings should default to true")
	}
	if !a.HasPerLayerInputs() {
		t.Error("default hidden_size_per_layer_input should activate per-layer inputs")
	}
	// Default rotary: full attention base 1000000 partial 0.25, sliding base 10000.
	checkFloat(t, "full rope theta", a.RopeTheta[fullAttention], 1000000.0)
	checkFloat(t, "full partial", a.RopePartial[fullAttention], 0.25)
	checkFloat(t, "sliding rope theta", a.RopeTheta[slidingAttention], 10000.0)
	// The last layer of the default pattern is full attention with the global head dim.
	last := a.NumLayers() - 1
	if !a.IsSliding(0) || a.IsSliding(last) {
		t.Error("default pattern should put sliding first and full last")
	}
	checkInt(t, "last head_dim", a.PerLayerHeadDim(last), 512)
	checkFloat(t, "embed_scale", a.EmbedScale(), 16.0)
}

func TestParseGemma4Errors(t *testing.T) {
	cases := []struct {
		name string
		cfg  string
	}{
		{"bad_json", `{`},
		{"no_heads", `{"hidden_size":16,"num_hidden_layers":2,"vocab_size":10,"sliding_window_pattern":2}`},
		{"bad_layer_type", `{"hidden_size":16,"num_attention_heads":4,"vocab_size":10,"layer_types":["bogus"]}`},
		{"sliding_no_window", `{"hidden_size":16,"num_attention_heads":4,"vocab_size":10,"layer_types":["sliding_attention"]}`},
		{"too_many_shared", `{"hidden_size":16,"num_attention_heads":4,"vocab_size":10,"num_kv_shared_layers":9,"layer_types":["full_attention"]}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := ParseGemma4TextArgs([]byte(c.cfg)); err == nil {
				t.Error("expected error")
			}
		})
	}
}

func TestGemma4PerLayerInputScales(t *testing.T) {
	a := &Gemma4TextArgs{HiddenSize: 256, HiddenSizePerLayerIn: 64}
	embed, gate, projection := a.PerLayerInputScales()
	checkFloat(t, "embed scale", embed, 8.0)             // sqrt(64)
	checkFloat(t, "gate scale", gate, math.Pow(2, -0.5)) // 2^-1/2
	checkFloat(t, "projection scale", projection, 1.0/16.0)
}

func TestGemma4Sanitize(t *testing.T) {
	arr := func() *mlxgo.Array {
		a, _ := mlxgo.NewFloat32([]float32{0}, 1)
		return a
	}
	tied := &Gemma4TextArgs{TieWordEmbeddings: true}
	w := map[string]*mlxgo.Array{
		"lm_head.weight": arr(),
		"model.layers.0.self_attn.rotary_emb.inv_freq": arr(),
		"model.layers.0.self_attn.q_proj.input_max":    arr(),
		"model.layers.0.self_attn.q_proj.output_min":   arr(),
		"model.norm.weight":                            arr(),
	}
	tied.Sanitize(w)
	for _, gone := range []string{
		"lm_head.weight",
		"model.layers.0.self_attn.rotary_emb.inv_freq",
		"model.layers.0.self_attn.q_proj.input_max",
		"model.layers.0.self_attn.q_proj.output_min",
	} {
		if _, ok := w[gone]; ok {
			t.Errorf("Sanitize should have dropped %q", gone)
		}
	}
	if _, ok := w["model.norm.weight"]; !ok {
		t.Error("Sanitize dropped an unrelated weight")
	}

	untied := &Gemma4TextArgs{TieWordEmbeddings: false}
	w2 := map[string]*mlxgo.Array{"lm_head.weight": arr()}
	untied.Sanitize(w2)
	if _, ok := w2["lm_head.weight"]; !ok {
		t.Error("untied Sanitize must keep lm_head.weight")
	}
}

func BenchmarkParseGemma4Args(b *testing.B) {
	b.ReportAllocs()
	cfg := []byte(`{"model_type":"gemma4_text","hidden_size":1536,"num_hidden_layers":35,` +
		`"intermediate_size":6144,"num_attention_heads":8,"head_dim":256,"global_head_dim":512,` +
		`"rms_norm_eps":1e-6,"vocab_size":262144,"num_key_value_heads":1,"num_kv_shared_layers":20,` +
		`"hidden_size_per_layer_input":256,"sliding_window":512,"sliding_window_pattern":5,` +
		`"final_logit_softcapping":30.0,"tie_word_embeddings":true}`)
	for b.Loop() {
		a, err := ParseGemma4TextArgs(cfg)
		if err != nil {
			b.Fatal(err)
		}
		_ = a.WeightNames()
		_ = a.PreviousKVs()
		_ = a.MakeCache()
	}
}
