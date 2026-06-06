// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// The fixture is captured from the reference dense GLM-4 model: args from
// ModelArgs.from_dict, the partial-rotary dims and derived attention widths, the
// weight names from the flattened parameter tree, and the cache kinds from
// make_prompt_cache.
type glm4Fixture struct {
	Derivations []struct {
		Label            string          `json:"label"`
		Config           json.RawMessage `json:"config"`
		HeadDim          int             `json:"head_dim"`
		NumLayers        int             `json:"num_layers"`
		NumKeyValueHeads int             `json:"num_key_value_heads"`
		RopeDims         int             `json:"rope_dims"`
		RopeTraditional  bool            `json:"rope_traditional"`
		Scale            float64         `json:"scale"`
		QProjOut         int             `json:"q_proj_out"`
		KVProjOut        int             `json:"kv_proj_out"`
		GQARepeat        int             `json:"gqa_repeat"`
		AttentionBias    bool            `json:"attention_bias"`
		WeightNames      []string        `json:"weight_names"`
		CacheCount       int             `json:"cache_count"`
		CacheTypes       []string        `json:"cache_types"`
	} `json:"derivations"`
}

func loadGlm4Fixture(t *testing.T) glm4Fixture {
	t.Helper()
	b, err := os.ReadFile("../testdata/glm4_args.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f glm4Fixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return f
}

func TestGlm4ArgsParity(t *testing.T) {
	f := loadGlm4Fixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParseGlm4Args(d.Config)
			if err != nil {
				t.Fatalf("ParseGlm4Args: %v", err)
			}
			checkInt(t, "head_dim", a.HeadDim, d.HeadDim)
			checkInt(t, "num_layers", a.NumLayers(), d.NumLayers)
			checkInt(t, "num_key_value_heads", a.NumKeyValueHeads, d.NumKeyValueHeads)
			checkInt(t, "rope_dims", a.RopeDims(), d.RopeDims)
			checkInt(t, "q_proj_out", a.QProjOut(), d.QProjOut)
			checkInt(t, "kv_proj_out", a.KVProjOut(), d.KVProjOut)
			checkInt(t, "gqa_repeat", a.GQARepeat(), d.GQARepeat)
			checkFloat(t, "scale", a.Scale(), d.Scale)
			if a.RopeTraditional != d.RopeTraditional {
				t.Errorf("rope_traditional = %v, want %v", a.RopeTraditional, d.RopeTraditional)
			}
			if a.AttentionBias != d.AttentionBias {
				t.Errorf("attention_bias = %v, want %v", a.AttentionBias, d.AttentionBias)
			}
		})
	}
}

func TestGlm4WeightNamesParity(t *testing.T) {
	f := loadGlm4Fixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParseGlm4Args(d.Config)
			if err != nil {
				t.Fatalf("ParseGlm4Args: %v", err)
			}
			got := a.WeightNames()
			if !reflect.DeepEqual(got, d.WeightNames) {
				t.Errorf("weight names mismatch (%d vs %d)\n got %v\nwant %v",
					len(got), len(d.WeightNames), got, d.WeightNames)
			}
		})
	}
}

func TestGlm4MakeCacheParity(t *testing.T) {
	f := loadGlm4Fixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParseGlm4Args(d.Config)
			if err != nil {
				t.Fatalf("ParseGlm4Args: %v", err)
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

func TestParseGlm4Defaults(t *testing.T) {
	// head_dim, rope_traditional, and max_position_embeddings absent: head_dim
	// derives from the head count, rope is traditional, max pos is 32768.
	cfg := `{"model_type":"glm4","hidden_size":32,"num_hidden_layers":2,"intermediate_size":64,` +
		`"num_attention_heads":4,"attention_bias":false,"rms_norm_eps":1e-6,"vocab_size":40,` +
		`"num_key_value_heads":2,"partial_rotary_factor":0.5,"rope_theta":10000.0}`
	a, err := ParseGlm4Args([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseGlm4Args: %v", err)
	}
	checkInt(t, "head_dim", a.HeadDim, 8)
	checkInt(t, "rope_dims", a.RopeDims(), 4)
	checkInt(t, "max_position_embeddings", a.MaxPositionEmbeddings, 32768)
	if !a.RopeTraditional {
		t.Error("rope_traditional should default to true")
	}
}

func TestParseGlm4Errors(t *testing.T) {
	cases := []struct {
		name string
		cfg  string
	}{
		{"bad_json", `{`},
		{"no_heads", `{"hidden_size":16,"num_hidden_layers":2,"vocab_size":10,"num_key_value_heads":2,"partial_rotary_factor":0.5}`},
		{"bad_gqa", `{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,"head_dim":4,"vocab_size":10,"num_key_value_heads":3,"partial_rotary_factor":0.5}`},
		{"bad_partial_hi", `{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,"head_dim":4,"vocab_size":10,"num_key_value_heads":2,"partial_rotary_factor":1.5}`},
		{"bad_partial_lo", `{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,"head_dim":4,"vocab_size":10,"num_key_value_heads":2,"partial_rotary_factor":0}`},
		{"no_layers", `{"hidden_size":16,"num_attention_heads":4,"head_dim":4,"vocab_size":10,"num_key_value_heads":2,"partial_rotary_factor":0.5}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := ParseGlm4Args([]byte(c.cfg)); err == nil {
				t.Error("expected error")
			}
		})
	}
}

// tinyGlm4Args is a 2-layer config for assembly tests.
func tinyGlm4Args(t *testing.T, bias bool) *Glm4Args {
	t.Helper()
	cfg := `{"model_type":"glm4","hidden_size":8,"num_hidden_layers":2,"intermediate_size":16,` +
		`"num_attention_heads":4,"attention_bias":` + boolStr(bias) + `,"head_dim":4,"rms_norm_eps":1e-6,` +
		`"vocab_size":32,"num_key_value_heads":2,"partial_rotary_factor":0.5,"rope_theta":10000.0}`
	a, err := ParseGlm4Args([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseGlm4Args: %v", err)
	}
	return a
}

func dummyGlm4Weights(t *testing.T, a *Glm4Args) map[string]*mlxgo.Array {
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

func TestNewGlm4ModelWiresWeights(t *testing.T) {
	for _, bias := range []bool{false, true} {
		a := tinyGlm4Args(t, bias)
		m, err := NewGlm4Model(a, dummyGlm4Weights(t, a))
		if err != nil {
			t.Fatalf("NewGlm4Model(bias=%v): %v", bias, err)
		}
		if len(m.layers) != a.NumLayers() {
			t.Errorf("layers = %d, want %d", len(m.layers), a.NumLayers())
		}
		if m.lmHead == nil {
			t.Error("GLM-4 is always untied: lmHead must be wired")
		}
		hasBias := m.layers[0].qBias != nil
		if hasBias != bias {
			t.Errorf("qBias present = %v, want %v", hasBias, bias)
		}
	}
}

func TestNewGlm4ModelMissingWeight(t *testing.T) {
	a := tinyGlm4Args(t, false)
	w := dummyGlm4Weights(t, a)
	delete(w, "lm_head.weight")
	if _, err := NewGlm4Model(a, w); err == nil {
		t.Error("expected missing-weight error for absent lm_head")
	}
}

func TestGlm4ForwardGracefulWithoutBackend(t *testing.T) {
	for _, bias := range []bool{false, true} {
		a := tinyGlm4Args(t, bias)
		m, err := NewGlm4Model(a, dummyGlm4Weights(t, a))
		if err != nil {
			t.Fatalf("NewGlm4Model: %v", err)
		}
		caches := make([]*KVTensorCache, a.NumLayers())
		for i := range caches {
			caches[i] = &KVTensorCache{}
		}
		if _, err := m.Forward([]int32{1, 2, 3}, caches, mlxgo.DefaultStream()); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
			t.Errorf("Forward(bias=%v) err = %v, want ErrMLXUnavailable", bias, err)
		}
	}
}

func TestGlm4ForwardCacheCountGuard(t *testing.T) {
	a := tinyGlm4Args(t, false)
	m, err := NewGlm4Model(a, dummyGlm4Weights(t, a))
	if err != nil {
		t.Fatalf("NewGlm4Model: %v", err)
	}
	if _, err := m.Forward([]int32{1}, []*KVTensorCache{{}}, mlxgo.DefaultStream()); err == nil {
		t.Error("expected a cache-count mismatch error")
	}
}

// tinyGlm4QuantArgs is the assembly config with a top-level affine quantization
// block, so the loader runs the fused projections through the packed path.
func tinyGlm4QuantArgs(t *testing.T, bias bool) *Glm4Args {
	t.Helper()
	cfg := `{"model_type":"glm4","hidden_size":8,"num_hidden_layers":2,"intermediate_size":16,` +
		`"num_attention_heads":4,"attention_bias":` + boolStr(bias) + `,"head_dim":4,"rms_norm_eps":1e-6,` +
		`"vocab_size":32,"num_key_value_heads":2,"partial_rotary_factor":0.5,"rope_theta":10000.0,` +
		`"quantization":{"group_size":64,"bits":4,"mode":"affine"}}`
	a, err := ParseGlm4Args([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseGlm4Args: %v", err)
	}
	return a
}

func dummyGlm4QuantWeights(t *testing.T, a *Glm4Args) map[string]*mlxgo.Array {
	t.Helper()
	w := dummyGlm4Weights(t, a)
	for _, name := range a.WeightNames() {
		if !quantizableModule(name) {
			continue
		}
		base := name[:len(name)-len(".weight")]
		for _, comp := range []string{".scales", ".biases"} {
			arr, err := mlxgo.NewFloat32([]float32{0}, 1)
			if err != nil {
				t.Fatalf("NewFloat32: %v", err)
			}
			w[base+comp] = arr
		}
	}
	return w
}

func TestParseGlm4ArgsCapturesQuantization(t *testing.T) {
	a := tinyGlm4QuantArgs(t, true)
	if a.quant != (quantConfig{GroupSize: 64, Bits: 4}) {
		t.Errorf("quant = %+v, want {64 4}", a.quant)
	}
	if d := tinyGlm4Args(t, true).quant; d.quantized() {
		t.Errorf("dense config reported quantized: %+v", d)
	}
}

func TestNewGlm4ModelQuantizedWiring(t *testing.T) {
	for _, bias := range []bool{false, true} {
		a := tinyGlm4QuantArgs(t, bias)
		m, err := NewGlm4Model(a, dummyGlm4QuantWeights(t, a))
		if err != nil {
			t.Fatalf("NewGlm4Model(bias=%v): %v", bias, err)
		}
		if !m.embedTokens.isQuantized() || !m.lmHead.isQuantized() {
			t.Error("embedding / head not loaded quantized")
		}
		for i := range m.layers {
			l := &m.layers[i]
			// The fused gate-up projection is one quantized module; the down
			// projection and the four attention projections are the rest.
			for _, q := range []*qLinear{l.qProj, l.kProj, l.vProj, l.oProj, l.gateUpProj, l.downProj} {
				if !q.isQuantized() {
					t.Fatalf("layer %d projection not loaded quantized", i)
				}
				if q.groupSize != 64 || q.bits != 4 {
					t.Fatalf("layer %d geometry = gs%d/b%d, want 64/4", i, q.groupSize, q.bits)
				}
			}
			// The additive qkv biases coexist with the affine packing.
			if hasBias := l.qBias != nil; hasBias != bias {
				t.Errorf("layer %d qBias present = %v, want %v", i, hasBias, bias)
			}
		}
	}
}

func TestNewGlm4ModelQuantizedMissingBiases(t *testing.T) {
	a := tinyGlm4QuantArgs(t, false)
	w := dummyGlm4QuantWeights(t, a)
	delete(w, "model.layers.0.mlp.gate_up_proj.biases")
	if _, err := NewGlm4Model(a, w); err == nil {
		t.Error("expected an error for a quantized module missing its biases")
	}
}

func TestGlm4QuantizedForwardReachesSeam(t *testing.T) {
	a := tinyGlm4QuantArgs(t, true)
	m, err := NewGlm4Model(a, dummyGlm4QuantWeights(t, a))
	if err != nil {
		t.Fatalf("NewGlm4Model: %v", err)
	}
	caches := make([]*KVTensorCache, a.NumLayers())
	for i := range caches {
		caches[i] = &KVTensorCache{}
	}
	if _, err := m.Forward([]int32{1, 2, 3}, caches, mlxgo.DefaultStream()); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Errorf("Forward err = %v, want ErrMLXUnavailable", err)
	}
}

func BenchmarkParseGlm4Args(b *testing.B) {
	b.ReportAllocs()
	cfg := []byte(`{"model_type":"glm4","hidden_size":4096,"num_hidden_layers":40,"intermediate_size":13696,` +
		`"num_attention_heads":32,"attention_bias":true,"head_dim":128,"rms_norm_eps":1e-5,"vocab_size":151552,` +
		`"num_key_value_heads":2,"partial_rotary_factor":0.5,"rope_theta":10000.0}`)
	for b.Loop() {
		a, err := ParseGlm4Args(cfg)
		if err != nil {
			b.Fatal(err)
		}
		_ = a.WeightNames()
	}
}
