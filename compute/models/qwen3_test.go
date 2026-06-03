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

// The fixture is produced from the reference Qwen3 model: the args records come
// from ModelArgs.from_dict, the weight names from the flattened, sanitized
// parameter tree, and the cache record from make_prompt_cache.
type qwen3Fixture struct {
	Args []struct {
		Label    string          `json:"label"`
		Config   json.RawMessage `json:"config"`
		Expected struct {
			ModelType             string  `json:"model_type"`
			HiddenSize            int     `json:"hidden_size"`
			NumHiddenLayers       int     `json:"num_hidden_layers"`
			IntermediateSize      int     `json:"intermediate_size"`
			NumAttentionHeads     int     `json:"num_attention_heads"`
			RMSNormEps            float64 `json:"rms_norm_eps"`
			VocabSize             int     `json:"vocab_size"`
			NumKeyValueHeads      int     `json:"num_key_value_heads"`
			MaxPositionEmbeddings int     `json:"max_position_embeddings"`
			RopeTheta             float64 `json:"rope_theta"`
			HeadDim               int     `json:"head_dim"`
			TieWordEmbeddings     bool    `json:"tie_word_embeddings"`
			Scale                 float64 `json:"scale"`
			QProjOut              int     `json:"q_proj_out"`
			KVProjOut             int     `json:"kv_proj_out"`
			GQARepeat             int     `json:"gqa_repeat"`
		} `json:"expected"`
	} `json:"args"`
	WeightNames []struct {
		Label  string          `json:"label"`
		Config json.RawMessage `json:"config"`
		Tie    bool            `json:"tie"`
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

func loadQwen3Fixture(t *testing.T) qwen3Fixture {
	t.Helper()
	b, err := os.ReadFile("../testdata/qwen3_args.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f qwen3Fixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return f
}

func TestQwen3ArgsParity(t *testing.T) {
	f := loadQwen3Fixture(t)
	for _, c := range f.Args {
		t.Run(c.Label, func(t *testing.T) {
			a, err := ParseQwen3Args(c.Config)
			if err != nil {
				t.Fatalf("ParseQwen3Args: %v", err)
			}
			e := c.Expected
			checkInt(t, "hidden_size", a.HiddenSize, e.HiddenSize)
			checkInt(t, "num_hidden_layers", a.NumHiddenLayers, e.NumHiddenLayers)
			checkInt(t, "intermediate_size", a.IntermediateSize, e.IntermediateSize)
			checkInt(t, "num_attention_heads", a.NumAttentionHeads, e.NumAttentionHeads)
			checkInt(t, "vocab_size", a.VocabSize, e.VocabSize)
			checkInt(t, "num_key_value_heads", a.NumKeyValueHeads, e.NumKeyValueHeads)
			checkInt(t, "max_position_embeddings", a.MaxPositionEmbeddings, e.MaxPositionEmbeddings)
			checkInt(t, "head_dim", a.HeadDim, e.HeadDim)
			if a.ModelType != e.ModelType {
				t.Errorf("model_type = %q, want %q", a.ModelType, e.ModelType)
			}
			if a.TieWordEmbeddings != e.TieWordEmbeddings {
				t.Errorf("tie_word_embeddings = %v, want %v", a.TieWordEmbeddings, e.TieWordEmbeddings)
			}
			checkFloat(t, "rms_norm_eps", a.RMSNormEps, e.RMSNormEps)
			checkFloat(t, "rope_theta", a.RopeTheta, e.RopeTheta)
			checkFloat(t, "scale", a.Scale(), e.Scale)
			checkInt(t, "q_proj_out", a.QProjOut(), e.QProjOut)
			checkInt(t, "kv_proj_out", a.KVProjOut(), e.KVProjOut)
			checkInt(t, "gqa_repeat", a.GQARepeat(), e.GQARepeat)
		})
	}
}

func TestQwen3WeightNamesParity(t *testing.T) {
	f := loadQwen3Fixture(t)
	for _, c := range f.WeightNames {
		t.Run(c.Label, func(t *testing.T) {
			a, err := ParseQwen3Args(c.Config)
			if err != nil {
				t.Fatalf("ParseQwen3Args: %v", err)
			}
			got := a.WeightNames()
			if !reflect.DeepEqual(got, c.Names) {
				t.Errorf("weight names mismatch\n got %v\nwant %v", got, c.Names)
			}
		})
	}
}

func TestQwen3MakeCacheParity(t *testing.T) {
	f := loadQwen3Fixture(t)
	for _, c := range f.Cache {
		t.Run(c.Label, func(t *testing.T) {
			a, err := ParseQwen3Args(c.Config)
			if err != nil {
				t.Fatalf("ParseQwen3Args: %v", err)
			}
			caches := a.MakeCache()
			if len(caches) != c.Expected.Count {
				t.Errorf("cache count = %d, want %d", len(caches), c.Expected.Count)
			}
			for i, kv := range caches {
				if kv == nil {
					t.Fatalf("cache[%d] is nil", i)
				}
				if !kv.Empty() || kv.Size() != 0 {
					t.Errorf("cache[%d] not fresh: empty=%v size=%d", i, kv.Empty(), kv.Size())
				}
			}
		})
	}
}

func TestQwen3SanitizeDropsTiedHead(t *testing.T) {
	arr := func() *mlxgo.Array {
		a, _ := mlxgo.NewFloat32([]float32{0}, 1)
		return a
	}
	tied := &Qwen3Args{TieWordEmbeddings: true}
	w := map[string]*mlxgo.Array{"lm_head.weight": arr(), "model.norm.weight": arr()}
	tied.Sanitize(w)
	if _, ok := w["lm_head.weight"]; ok {
		t.Error("tied Sanitize should drop lm_head.weight")
	}
	if _, ok := w["model.norm.weight"]; !ok {
		t.Error("Sanitize dropped an unrelated weight")
	}

	untied := &Qwen3Args{TieWordEmbeddings: false}
	w2 := map[string]*mlxgo.Array{"lm_head.weight": arr()}
	untied.Sanitize(w2)
	if _, ok := w2["lm_head.weight"]; !ok {
		t.Error("untied Sanitize must keep lm_head.weight")
	}
}

func TestParseQwen3ArgsErrors(t *testing.T) {
	cases := []struct {
		name string
		cfg  string
	}{
		{"bad_json", `{`},
		{"no_heads", `{"hidden_size":16,"num_hidden_layers":2,"head_dim":4,"vocab_size":10}`},
		{"no_hidden", `{"num_attention_heads":4,"num_hidden_layers":2,"head_dim":4,"vocab_size":10}`},
		{"no_layers", `{"hidden_size":16,"num_attention_heads":4,"head_dim":4,"vocab_size":10}`},
		{"no_vocab", `{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,"head_dim":4}`},
		{"gqa_not_multiple", `{"hidden_size":16,"num_attention_heads":6,"num_key_value_heads":4,"num_hidden_layers":2,"head_dim":4,"vocab_size":10}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := ParseQwen3Args([]byte(c.cfg)); err == nil {
				t.Error("expected error")
			}
		})
	}
}

func TestParseQwen3ArgsHeadDimDefault(t *testing.T) {
	// head_dim absent: the reference derives hidden_size / num_attention_heads.
	cfg := `{"hidden_size":32,"num_attention_heads":4,"num_hidden_layers":2,"vocab_size":10}`
	a, err := ParseQwen3Args([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseQwen3Args: %v", err)
	}
	if a.HeadDim != 8 {
		t.Errorf("head_dim default = %d, want 8", a.HeadDim)
	}
	if a.NumKeyValueHeads != 4 {
		t.Errorf("num_key_value_heads default = %d, want 4", a.NumKeyValueHeads)
	}
}

func checkInt(t *testing.T, name string, got, want int) {
	t.Helper()
	if got != want {
		t.Errorf("%s = %d, want %d", name, got, want)
	}
}

func checkFloat(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("%s = %v, want %v", name, got, want)
	}
}

func BenchmarkParseQwen3Args(b *testing.B) {
	b.ReportAllocs()
	cfg := []byte(`{"model_type":"qwen3","hidden_size":1024,"num_hidden_layers":28,"intermediate_size":3072,"num_attention_heads":16,"rms_norm_eps":1e-6,"vocab_size":151936,"num_key_value_heads":8,"max_position_embeddings":40960,"rope_theta":1000000.0,"head_dim":128,"tie_word_embeddings":true}`)
	for b.Loop() {
		a, err := ParseQwen3Args(cfg)
		if err != nil {
			b.Fatal(err)
		}
		_ = a.WeightNames()
	}
}
