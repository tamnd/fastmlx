// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// The fixture is captured from the reference phi3 model: args from
// ModelArgs.from_dict (with __post_init__ resolving num_kv_heads and the
// rope_scaling validation), the fused-qkv band geometry, the weight names from
// the flattened parameter tree, the cache kinds from make_prompt_cache, and the
// SuScaledRoPE frequencies and magnitude scale for the longrope path.
type phi4Fixture struct {
	Derivations []struct {
		Label             string          `json:"label"`
		Config            json.RawMessage `json:"config"`
		NumKeyValueHeads  int             `json:"num_key_value_heads"`
		HeadDim           int             `json:"head_dim"`
		Scale             float64         `json:"scale"`
		RopeDim           int             `json:"rope_dim"`
		OpSize            int             `json:"op_size"`
		QueryPos          int             `json:"query_pos"`
		KVSize            int             `json:"kv_size"`
		RopeScalingKept   bool            `json:"rope_scaling_kept"`
		TieWordEmbeddings bool            `json:"tie_word_embeddings"`
		WeightNames       []string        `json:"weight_names"`
		CacheCount        int             `json:"cache_count"`
		CacheTypes        []string        `json:"cache_types"`
	} `json:"derivations"`
	SuCases []struct {
		Label                         string    `json:"label"`
		Dims                          int       `json:"dims"`
		Base                          float64   `json:"base"`
		MaxPositionEmbeddings         int       `json:"max_position_embeddings"`
		OriginalMaxPositionEmbeddings int       `json:"original_max_position_embeddings"`
		LongFactor                    any       `json:"long_factor"`
		Freqs                         []float64 `json:"freqs"`
		Scale                         float64   `json:"scale"`
	} `json:"su_cases"`
}

func loadPhi4Fixture(t *testing.T) phi4Fixture {
	t.Helper()
	b, err := os.ReadFile("../testdata/phi4_args.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f phi4Fixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return f
}

func TestPhi4ArgsParity(t *testing.T) {
	f := loadPhi4Fixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParsePhi4Args(d.Config)
			if err != nil {
				t.Fatalf("ParsePhi4Args: %v", err)
			}
			checkInt(t, "num_key_value_heads", a.NumKeyValueHeads, d.NumKeyValueHeads)
			checkInt(t, "head_dim", a.HeadDim(), d.HeadDim)
			checkInt(t, "rope_dim", a.RopeDims(), d.RopeDim)
			checkInt(t, "op_size", a.OpSize(), d.OpSize)
			checkInt(t, "query_pos", a.QueryPos(), d.QueryPos)
			checkInt(t, "kv_size", a.KVSize(), d.KVSize)
			checkFloat(t, "scale", a.Scale(), d.Scale)
			if (a.RopeScaling != nil) != d.RopeScalingKept {
				t.Errorf("rope_scaling kept = %v, want %v", a.RopeScaling != nil, d.RopeScalingKept)
			}
			if a.TieWordEmbeddings != d.TieWordEmbeddings {
				t.Errorf("tie = %v, want %v", a.TieWordEmbeddings, d.TieWordEmbeddings)
			}
		})
	}
}

func TestPhi4WeightNamesParity(t *testing.T) {
	f := loadPhi4Fixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParsePhi4Args(d.Config)
			if err != nil {
				t.Fatalf("ParsePhi4Args: %v", err)
			}
			got := a.WeightNames()
			if !reflect.DeepEqual(got, d.WeightNames) {
				t.Errorf("weight names mismatch (%d vs %d)\n got %v\nwant %v",
					len(got), len(d.WeightNames), got, d.WeightNames)
			}
		})
	}
}

func TestPhi4MakeCacheParity(t *testing.T) {
	f := loadPhi4Fixture(t)
	for _, d := range f.Derivations {
		t.Run(d.Label, func(t *testing.T) {
			a, err := ParsePhi4Args(d.Config)
			if err != nil {
				t.Fatalf("ParsePhi4Args: %v", err)
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

func TestPhi4SuRopeParity(t *testing.T) {
	f := loadPhi4Fixture(t)
	for _, c := range f.SuCases {
		t.Run(c.Label, func(t *testing.T) {
			// Drive the SuScaledRoPE helpers directly with the captured inputs.
			rs := &Phi4RopeScaling{Type: "longrope", OriginalMaxPositionEmbeddings: c.OriginalMaxPositionEmbeddings}
			switch lf := c.LongFactor.(type) {
			case []any:
				for _, v := range lf {
					rs.LongFactor = append(rs.LongFactor, v.(float64))
				}
			case float64:
				rs.LongFactor = []float64{lf}
			}
			a := &Phi4Args{
				HiddenSize:            c.Dims,
				NumAttentionHeads:     1,
				RopeTheta:             c.Base,
				PartialRotaryFactor:   1.0,
				MaxPositionEmbeddings: c.MaxPositionEmbeddings,
				RopeScaling:           rs,
			}
			freqs := a.SuRopeFreqs()
			if len(freqs) != len(c.Freqs) {
				t.Fatalf("freq count = %d, want %d", len(freqs), len(c.Freqs))
			}
			// The reference computes the frequencies in float32; compare at a
			// float32-relative tolerance rather than the exact float64 epsilon.
			for i := range freqs {
				if rel := math.Abs(freqs[i]-c.Freqs[i]) / math.Abs(c.Freqs[i]); rel > 1e-5 {
					t.Errorf("freq[%d] = %v, want %v (rel %g)", i, freqs[i], c.Freqs[i], rel)
				}
			}
			// The magnitude scale is pure Python float64 math, so it matches exactly.
			checkFloat(t, "scale", a.SuRopeScale(), c.Scale)
		})
	}
}

func TestParsePhi4Defaults(t *testing.T) {
	// num_key_value_heads, rope_theta, partial_rotary_factor, and the max-position
	// fields are all absent: kv heads fall back to the head count, rope_theta to
	// 10000, partial factor to 1.0, and the head is untied.
	cfg := `{"model_type":"phi3","hidden_size":32,"num_hidden_layers":2,"intermediate_size":64,` +
		`"num_attention_heads":4,"rms_norm_eps":1e-5,"vocab_size":40}`
	a, err := ParsePhi4Args([]byte(cfg))
	if err != nil {
		t.Fatalf("ParsePhi4Args: %v", err)
	}
	checkInt(t, "num_key_value_heads", a.NumKeyValueHeads, 4)
	checkFloat(t, "rope_theta", a.RopeTheta, 10000)
	checkFloat(t, "partial_rotary_factor", a.PartialRotaryFactor, 1.0)
	checkInt(t, "max_position_embeddings", a.MaxPositionEmbeddings, 131072)
	checkInt(t, "original_max_position_embeddings", a.OriginalMaxPositionEmbeddings, 4096)
	if a.TieWordEmbeddings {
		t.Error("tie should default to false")
	}
	if a.RopeScaling != nil {
		t.Error("rope_scaling should be nil when absent")
	}
}

func TestParsePhi4Errors(t *testing.T) {
	cases := []struct {
		name string
		cfg  string
	}{
		{"bad_json", `{`},
		{"no_heads", `{"hidden_size":16,"num_hidden_layers":2,"vocab_size":10}`},
		{"hidden_not_multiple", `{"hidden_size":18,"num_attention_heads":4,"num_hidden_layers":2,"vocab_size":10}`},
		{"bad_gqa", `{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,"vocab_size":10,"num_key_value_heads":3}`},
		{"bad_partial", `{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,"vocab_size":10,"partial_rotary_factor":1.5}`},
		{"no_layers", `{"hidden_size":16,"num_attention_heads":4,"vocab_size":10}`},
		{"rope_scaling_missing_keys", `{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,"vocab_size":10,"rope_scaling":{"type":"linear","factor":2.0}}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := ParsePhi4Args([]byte(c.cfg)); err == nil {
				t.Error("expected error")
			}
		})
	}
}

func TestParsePhi4UnsupportedRopeTypeDropped(t *testing.T) {
	// A type outside longrope/su/linear is dropped (the reference warns and
	// continues unscaled), so parsing succeeds with no scaling.
	cfg := `{"model_type":"phi3","hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,` +
		`"intermediate_size":32,"rms_norm_eps":1e-5,"vocab_size":10,` +
		`"rope_scaling":{"type":"mrope","long_factor":[1.0],"rope_type":"mrope"}}`
	a, err := ParsePhi4Args([]byte(cfg))
	if err != nil {
		t.Fatalf("ParsePhi4Args: %v", err)
	}
	if a.RopeScaling != nil {
		t.Error("unsupported rope type should be dropped to nil")
	}
}

func TestPhi4LinearRopeScale(t *testing.T) {
	cfg := `{"model_type":"phi3","hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,` +
		`"intermediate_size":32,"rms_norm_eps":1e-5,"vocab_size":10,` +
		`"rope_scaling":{"type":"linear","factor":4.0,"long_factor":[1.0]}}`
	a, err := ParsePhi4Args([]byte(cfg))
	if err != nil {
		t.Fatalf("ParsePhi4Args: %v", err)
	}
	checkFloat(t, "linear_rope_scale", a.LinearRopeScale(), 0.25)
	if a.UsesSuRope() {
		t.Error("linear scaling is not the Su path")
	}
}

// tinyPhi4Args is a 2-layer config for assembly tests.
func tinyPhi4Args(t *testing.T, tie bool) *Phi4Args {
	t.Helper()
	cfg := `{"model_type":"phi3","hidden_size":16,"num_hidden_layers":2,"intermediate_size":32,` +
		`"num_attention_heads":4,"rms_norm_eps":1e-5,"vocab_size":40,"num_key_value_heads":2,` +
		`"rope_theta":250000.0,"tie_word_embeddings":` + boolStr(tie) + `}`
	a, err := ParsePhi4Args([]byte(cfg))
	if err != nil {
		t.Fatalf("ParsePhi4Args: %v", err)
	}
	return a
}

func dummyPhi4Weights(t *testing.T, a *Phi4Args) map[string]*mlxgo.Array {
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

func TestNewPhi4ModelWiresWeights(t *testing.T) {
	for _, tie := range []bool{false, true} {
		a := tinyPhi4Args(t, tie)
		m, err := NewPhi4Model(a, dummyPhi4Weights(t, a))
		if err != nil {
			t.Fatalf("NewPhi4Model(tie=%v): %v", tie, err)
		}
		if len(m.layers) != a.NumLayers() {
			t.Errorf("layers = %d, want %d", len(m.layers), a.NumLayers())
		}
		if tie && m.lmHead != nil {
			t.Error("tied model must reuse the embedding (lmHead nil)")
		}
		if !tie && m.lmHead == nil {
			t.Error("untied model must wire lmHead")
		}
	}
}

func TestNewPhi4ModelMissingWeight(t *testing.T) {
	a := tinyPhi4Args(t, false)
	w := dummyPhi4Weights(t, a)
	delete(w, "model.layers.0.self_attn.qkv_proj.weight")
	if _, err := NewPhi4Model(a, w); err == nil {
		t.Error("expected missing-weight error for absent qkv_proj")
	}
}

func TestPhi4ForwardGracefulWithoutBackend(t *testing.T) {
	for _, tie := range []bool{false, true} {
		a := tinyPhi4Args(t, tie)
		m, err := NewPhi4Model(a, dummyPhi4Weights(t, a))
		if err != nil {
			t.Fatalf("NewPhi4Model: %v", err)
		}
		caches := make([]*KVTensorCache, a.NumLayers())
		for i := range caches {
			caches[i] = &KVTensorCache{}
		}
		if _, err := m.Forward([]int32{1, 2, 3}, caches, mlxgo.DefaultStream()); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
			t.Errorf("Forward(tie=%v) err = %v, want ErrMLXUnavailable", tie, err)
		}
	}
}

func TestPhi4ForwardSuRopeSeam(t *testing.T) {
	// A longrope config must report the staged seam before any kernel runs, not
	// ErrMLXUnavailable.
	cfg := `{"model_type":"phi3","hidden_size":16,"num_hidden_layers":2,"intermediate_size":32,` +
		`"num_attention_heads":4,"rms_norm_eps":1e-5,"vocab_size":40,"num_key_value_heads":2,` +
		`"rope_theta":250000.0,"max_position_embeddings":131072,` +
		`"rope_scaling":{"type":"longrope","short_factor":[1.0,1.1],"long_factor":[1.0,1.4],` +
		`"original_max_position_embeddings":4096}}`
	a, err := ParsePhi4Args([]byte(cfg))
	if err != nil {
		t.Fatalf("ParsePhi4Args: %v", err)
	}
	if !a.UsesSuRope() {
		t.Fatal("expected the Su rope path")
	}
	m, err := NewPhi4Model(a, dummyPhi4Weights(t, a))
	if err != nil {
		t.Fatalf("NewPhi4Model: %v", err)
	}
	caches := make([]*KVTensorCache, a.NumLayers())
	for i := range caches {
		caches[i] = &KVTensorCache{}
	}
	_, err = m.Forward([]int32{1, 2}, caches, mlxgo.DefaultStream())
	if err == nil || errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Errorf("expected the Su-rope seam error, got %v", err)
	}
}

func TestPhi4ForwardCacheCountGuard(t *testing.T) {
	a := tinyPhi4Args(t, false)
	m, err := NewPhi4Model(a, dummyPhi4Weights(t, a))
	if err != nil {
		t.Fatalf("NewPhi4Model: %v", err)
	}
	if _, err := m.Forward([]int32{1}, []*KVTensorCache{{}}, mlxgo.DefaultStream()); err == nil {
		t.Error("expected a cache-count mismatch error")
	}
}

func BenchmarkParsePhi4Args(b *testing.B) {
	b.ReportAllocs()
	cfg := []byte(`{"model_type":"phi3","hidden_size":5120,"num_hidden_layers":40,"intermediate_size":17920,` +
		`"num_attention_heads":40,"rms_norm_eps":1e-5,"vocab_size":100352,"num_key_value_heads":10,` +
		`"rope_theta":250000.0}`)
	for b.Loop() {
		a, err := ParsePhi4Args(cfg)
		if err != nil {
			b.Fatal(err)
		}
		_ = a.WeightNames()
	}
}
