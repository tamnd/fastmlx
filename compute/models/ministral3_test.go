// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/tamnd/fastmlx/compute"
	"github.com/tamnd/fastmlx/mlxgo"
)

// The fixture is captured from the reference Ministral 3 text model: args from
// ModelArgs.from_dict, the attention-scale vectors from _get_llama_4_attn_scale,
// the weight names from the flattened sanitized parameter tree, and the cache
// kinds from make_cache.
type ministralFixture struct {
	Args []struct {
		Label    string          `json:"label"`
		Config   json.RawMessage `json:"config"`
		Expected struct {
			HiddenSize                    int      `json:"hidden_size"`
			NumHiddenLayers               int      `json:"num_hidden_layers"`
			IntermediateSize              int      `json:"intermediate_size"`
			NumAttentionHeads             int      `json:"num_attention_heads"`
			VocabSize                     int      `json:"vocab_size"`
			NumKeyValueHeads              int      `json:"num_key_value_heads"`
			HeadDim                       int      `json:"head_dim"`
			RMSNormEps                    float64  `json:"rms_norm_eps"`
			TieWordEmbeddings             bool     `json:"tie_word_embeddings"`
			SlidingWindow                 int      `json:"sliding_window"`
			LayerTypes                    []string `json:"layer_types"`
			NumLayersEffective            int      `json:"num_layers_effective"`
			RopeTheta                     float64  `json:"rope_theta"`
			Llama4ScalingBeta             float64  `json:"llama_4_scaling_beta"`
			OriginalMaxPositionEmbeddings int      `json:"original_max_position_embeddings"`
			Scale                         float64  `json:"scale"`
			QProjOut                      int      `json:"q_proj_out"`
			KVProjOut                     int      `json:"kv_proj_out"`
			GQARepeat                     int      `json:"gqa_repeat"`
		} `json:"expected"`
	} `json:"args"`
	AttnScale []struct {
		Label                 string    `json:"label"`
		Size                  int       `json:"size"`
		Offset                int       `json:"offset"`
		Beta                  float64   `json:"beta"`
		MaxPositionEmbeddings int       `json:"max_position_embeddings"`
		Values                []float64 `json:"values"`
	} `json:"attn_scale"`
	WeightNames []struct {
		Label  string          `json:"label"`
		Config json.RawMessage `json:"config"`
		Names  []string        `json:"names"`
	} `json:"weight_names"`
	Cache []struct {
		Label    string          `json:"label"`
		Config   json.RawMessage `json:"config"`
		Expected struct {
			Count int      `json:"count"`
			Types []string `json:"types"`
		} `json:"expected"`
	} `json:"cache"`
}

func loadMinistralFixture(t *testing.T) ministralFixture {
	t.Helper()
	b, err := os.ReadFile("../testdata/ministral3_args.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f ministralFixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return f
}

func TestMinistralArgsParity(t *testing.T) {
	f := loadMinistralFixture(t)
	for _, c := range f.Args {
		t.Run(c.Label, func(t *testing.T) {
			a, err := ParseMinistralArgs(c.Config)
			if err != nil {
				t.Fatalf("ParseMinistralArgs: %v", err)
			}
			e := c.Expected
			checkInt(t, "hidden_size", a.HiddenSize, e.HiddenSize)
			checkInt(t, "num_attention_heads", a.NumAttentionHeads, e.NumAttentionHeads)
			checkInt(t, "vocab_size", a.VocabSize, e.VocabSize)
			checkInt(t, "num_key_value_heads", a.NumKeyValueHeads, e.NumKeyValueHeads)
			checkInt(t, "head_dim", a.HeadDim, e.HeadDim)
			checkInt(t, "sliding_window", a.SlidingWindow, e.SlidingWindow)
			checkInt(t, "num_layers_effective", a.NumLayers(), e.NumLayersEffective)
			checkInt(t, "original_max_position_embeddings", a.OriginalMaxPositionEmbeddings, e.OriginalMaxPositionEmbeddings)
			if !reflect.DeepEqual(a.LayerTypes, e.LayerTypes) {
				t.Errorf("layer_types = %v, want %v", a.LayerTypes, e.LayerTypes)
			}
			if a.TieWordEmbeddings != e.TieWordEmbeddings {
				t.Errorf("tie_word_embeddings = %v, want %v", a.TieWordEmbeddings, e.TieWordEmbeddings)
			}
			checkFloat(t, "rms_norm_eps", a.RMSNormEps, e.RMSNormEps)
			checkFloat(t, "rope_theta", a.RopeTheta, e.RopeTheta)
			checkFloat(t, "llama_4_scaling_beta", a.Llama4ScalingBeta, e.Llama4ScalingBeta)
			checkFloat(t, "scale", a.Scale(), e.Scale)
			checkInt(t, "q_proj_out", a.QProjOut(), e.QProjOut)
			checkInt(t, "kv_proj_out", a.KVProjOut(), e.KVProjOut)
			checkInt(t, "gqa_repeat", a.GQARepeat(), e.GQARepeat)
		})
	}
}

func TestMinistralAttnScaleParity(t *testing.T) {
	f := loadMinistralFixture(t)
	for _, c := range f.AttnScale {
		t.Run(c.Label, func(t *testing.T) {
			a := &MinistralArgs{Llama4ScalingBeta: c.Beta, OriginalMaxPositionEmbeddings: c.MaxPositionEmbeddings}
			got := a.AttnScale(c.Size, c.Offset)
			if len(got) != len(c.Values) {
				t.Fatalf("len = %d, want %d", len(got), len(c.Values))
			}
			for i := range got {
				if math.Abs(float64(got[i])-c.Values[i]) > 1e-5 {
					t.Errorf("attn_scale[%d] = %v, want %v", i, got[i], c.Values[i])
				}
			}
		})
	}
}

func TestMinistralWeightNamesParity(t *testing.T) {
	f := loadMinistralFixture(t)
	for _, c := range f.WeightNames {
		t.Run(c.Label, func(t *testing.T) {
			a, err := ParseMinistralArgs(c.Config)
			if err != nil {
				t.Fatalf("ParseMinistralArgs: %v", err)
			}
			got := a.WeightNames()
			if !reflect.DeepEqual(got, c.Names) {
				t.Errorf("weight names mismatch\n got %v\nwant %v", got, c.Names)
			}
		})
	}
}

func TestMinistralMakeCacheParity(t *testing.T) {
	f := loadMinistralFixture(t)
	for _, c := range f.Cache {
		t.Run(c.Label, func(t *testing.T) {
			a, err := ParseMinistralArgs(c.Config)
			if err != nil {
				t.Fatalf("ParseMinistralArgs: %v", err)
			}
			caches := a.MakeCache()
			if len(caches) != c.Expected.Count {
				t.Fatalf("cache count = %d, want %d", len(caches), c.Expected.Count)
			}
			for i, kv := range caches {
				got := cacheTypeName(kv)
				if got != c.Expected.Types[i] {
					t.Errorf("cache[%d] type = %s, want %s", i, got, c.Expected.Types[i])
				}
			}
		})
	}
}

func cacheTypeName(c compute.Cache) string {
	switch c.(type) {
	case *compute.KVCache:
		return "KVCache"
	case *compute.RotatingKVCache:
		return "RotatingKVCache"
	default:
		return "unknown"
	}
}

func TestParseMinistralArgsDefaults(t *testing.T) {
	// layer_types, num_key_value_heads, head_dim, and tie_word_embeddings absent:
	// the reference derives head_dim and kv heads, ties the embeddings, and lays
	// out num_hidden_layers full-attention layers.
	cfg := `{"hidden_size":32,"num_attention_heads":4,"num_hidden_layers":3,"vocab_size":10,` +
		`"rope_parameters":{"rope_theta":10000.0}}`
	a, err := ParseMinistralArgs([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseMinistralArgs: %v", err)
	}
	checkInt(t, "head_dim", a.HeadDim, 8)
	checkInt(t, "num_key_value_heads", a.NumKeyValueHeads, 4)
	checkInt(t, "num_layers", a.NumLayers(), 3)
	if !a.TieWordEmbeddings {
		t.Error("tie_word_embeddings should default to true")
	}
	for i := range a.LayerTypes {
		if a.IsSliding(i) {
			t.Errorf("layer %d should default to full attention", i)
		}
	}
	// With no llama4 beta the scale collapses to ones.
	for i, v := range a.AttnScale(4, 0) {
		if v != 1 {
			t.Errorf("attn_scale[%d] = %v, want 1 without a beta", i, v)
		}
	}
}

func TestParseMinistralArgsErrors(t *testing.T) {
	cases := []struct {
		name string
		cfg  string
	}{
		{"bad_json", `{`},
		{"no_rope", `{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,"head_dim":4,"vocab_size":10}`},
		{"no_theta", `{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,"head_dim":4,"vocab_size":10,"rope_parameters":{}}`},
		{"no_heads", `{"hidden_size":16,"num_hidden_layers":2,"head_dim":4,"vocab_size":10,"rope_parameters":{"rope_theta":1.0}}`},
		{"sliding_no_window", `{"hidden_size":16,"num_attention_heads":4,"head_dim":4,"vocab_size":10,"layer_types":["sliding_attention"],"rope_parameters":{"rope_theta":1.0}}`},
		{"bad_layer_type", `{"hidden_size":16,"num_attention_heads":4,"head_dim":4,"vocab_size":10,"layer_types":["bogus"],"rope_parameters":{"rope_theta":1.0}}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := ParseMinistralArgs([]byte(c.cfg)); err == nil {
				t.Error("expected error")
			}
		})
	}
}

func TestMinistralSanitize(t *testing.T) {
	arr := func() *mlxgo.Array {
		a, _ := mlxgo.NewFloat32([]float32{0}, 1)
		return a
	}
	tied := &MinistralArgs{TieWordEmbeddings: true}
	w := map[string]*mlxgo.Array{
		"lm_head.weight": arr(),
		"model.layers.0.self_attn.rotary_emb.inv_freq":  arr(),
		"model.layers.0.mlp.gate_proj.activation_scale": arr(),
		"model.norm.weight":                             arr(),
	}
	tied.Sanitize(w)
	for _, gone := range []string{"lm_head.weight", "model.layers.0.self_attn.rotary_emb.inv_freq", "model.layers.0.mlp.gate_proj.activation_scale"} {
		if _, ok := w[gone]; ok {
			t.Errorf("Sanitize should have dropped %q", gone)
		}
	}
	if _, ok := w["model.norm.weight"]; !ok {
		t.Error("Sanitize dropped an unrelated weight")
	}

	untied := &MinistralArgs{TieWordEmbeddings: false}
	w2 := map[string]*mlxgo.Array{"lm_head.weight": arr()}
	untied.Sanitize(w2)
	if _, ok := w2["lm_head.weight"]; !ok {
		t.Error("untied Sanitize must keep lm_head.weight")
	}
}

// tinyMinistralArgs is a 2-layer (one full, one sliding) config for assembly.
func tinyMinistralArgs(t *testing.T, tie bool) *MinistralArgs {
	t.Helper()
	cfg := `{"model_type":"ministral3","hidden_size":8,"num_hidden_layers":2,"intermediate_size":16,` +
		`"num_attention_heads":4,"rms_norm_eps":1e-6,"vocab_size":32,"num_key_value_heads":2,` +
		`"max_position_embeddings":128,"sliding_window":4,` +
		`"layer_types":["full_attention","sliding_attention"],` +
		`"rope_parameters":{"rope_theta":100000000.0,"llama_4_scaling_beta":0.5,"original_max_position_embeddings":128},` +
		`"tie_word_embeddings":` + boolStr(tie) + `}`
	a, err := ParseMinistralArgs([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseMinistralArgs: %v", err)
	}
	return a
}

func dummyMinistralWeights(t *testing.T, a *MinistralArgs) map[string]*mlxgo.Array {
	t.Helper()
	w := make(map[string]*mlxgo.Array)
	for _, name := range a.WeightNames() {
		arr, err := mlxgo.NewFloat32([]float32{0}, 1)
		if err != nil {
			t.Fatalf("NewFloat32: %v", err)
		}
		w[name] = arr
	}
	return w
}

func TestNewMinistral3ModelWiresWeights(t *testing.T) {
	for _, tie := range []bool{true, false} {
		a := tinyMinistralArgs(t, tie)
		m, err := NewMinistral3Model(a, dummyMinistralWeights(t, a))
		if err != nil {
			t.Fatalf("NewMinistral3Model(tie=%v): %v", tie, err)
		}
		if len(m.layers) != a.NumLayers() {
			t.Errorf("layers = %d, want %d", len(m.layers), a.NumLayers())
		}
		if m.layers[0].sliding || !m.layers[1].sliding {
			t.Error("per-layer sliding flags not wired from layer_types")
		}
		if tie && m.lmHead != nil {
			t.Error("tied model should have a nil lmHead")
		}
		if !tie && m.lmHead == nil {
			t.Error("untied model should wire lmHead")
		}
	}
}

func TestMinistral3ForwardGracefulWithoutBackend(t *testing.T) {
	a := tinyMinistralArgs(t, true)
	m, err := NewMinistral3Model(a, dummyMinistralWeights(t, a))
	if err != nil {
		t.Fatalf("NewMinistral3Model: %v", err)
	}
	caches := make([]*KVTensorCache, a.NumLayers())
	for i := range caches {
		caches[i] = &KVTensorCache{}
	}
	if _, err := m.Forward([]int32{1, 2, 3}, caches, mlxgo.DefaultStream()); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Errorf("Forward err = %v, want ErrMLXUnavailable", err)
	}
}

func BenchmarkParseMinistralArgs(b *testing.B) {
	b.ReportAllocs()
	cfg := []byte(`{"model_type":"ministral3","hidden_size":4096,"num_hidden_layers":36,"intermediate_size":12288,` +
		`"num_attention_heads":32,"rms_norm_eps":1e-5,"vocab_size":131072,"num_key_value_heads":8,"head_dim":128,` +
		`"sliding_window":4096,"layer_types":["full_attention","sliding_attention"],` +
		`"rope_parameters":{"rope_theta":100000000.0,"llama_4_scaling_beta":0.5,"original_max_position_embeddings":8192},` +
		`"tie_word_embeddings":false}`)
	for b.Loop() {
		a, err := ParseMinistralArgs(cfg)
		if err != nil {
			b.Fatal(err)
		}
		_ = a.WeightNames()
	}
}
