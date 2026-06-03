// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// The fixture is captured from the reference Llama model: the args records come
// from ModelArgs.from_dict, the weight names from the flattened, sanitized
// parameter tree, and the cache record from make_prompt_cache.
type llamaFixture struct {
	Args []struct {
		Label    string          `json:"label"`
		Config   json.RawMessage `json:"config"`
		Expected struct {
			HiddenSize        int     `json:"hidden_size"`
			NumHiddenLayers   int     `json:"num_hidden_layers"`
			IntermediateSize  int     `json:"intermediate_size"`
			NumAttentionHeads int     `json:"num_attention_heads"`
			VocabSize         int     `json:"vocab_size"`
			NumKeyValueHeads  int     `json:"num_key_value_heads"`
			HeadDim           int     `json:"head_dim"`
			RopeTheta         float64 `json:"rope_theta"`
			RMSNormEps        float64 `json:"rms_norm_eps"`
			TieWordEmbeddings bool    `json:"tie_word_embeddings"`
			AttentionBias     bool    `json:"attention_bias"`
			MLPBias           bool    `json:"mlp_bias"`
			RopeTraditional   bool    `json:"rope_traditional"`
			Scale             float64 `json:"scale"`
			QProjOut          int     `json:"q_proj_out"`
			KVProjOut         int     `json:"kv_proj_out"`
			GQARepeat         int     `json:"gqa_repeat"`
		} `json:"expected"`
	} `json:"args"`
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

func loadLlamaFixture(t *testing.T) llamaFixture {
	t.Helper()
	b, err := os.ReadFile("../testdata/llama_args.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f llamaFixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return f
}

func TestLlamaArgsParity(t *testing.T) {
	f := loadLlamaFixture(t)
	for _, c := range f.Args {
		t.Run(c.Label, func(t *testing.T) {
			a, err := ParseLlamaArgs(c.Config)
			if err != nil {
				t.Fatalf("ParseLlamaArgs: %v", err)
			}
			e := c.Expected
			checkInt(t, "hidden_size", a.HiddenSize, e.HiddenSize)
			checkInt(t, "num_hidden_layers", a.NumHiddenLayers, e.NumHiddenLayers)
			checkInt(t, "intermediate_size", a.IntermediateSize, e.IntermediateSize)
			checkInt(t, "num_attention_heads", a.NumAttentionHeads, e.NumAttentionHeads)
			checkInt(t, "vocab_size", a.VocabSize, e.VocabSize)
			checkInt(t, "num_key_value_heads", a.NumKeyValueHeads, e.NumKeyValueHeads)
			checkInt(t, "head_dim", a.HeadDim, e.HeadDim)
			if a.TieWordEmbeddings != e.TieWordEmbeddings {
				t.Errorf("tie_word_embeddings = %v, want %v", a.TieWordEmbeddings, e.TieWordEmbeddings)
			}
			if a.AttentionBias != e.AttentionBias {
				t.Errorf("attention_bias = %v, want %v", a.AttentionBias, e.AttentionBias)
			}
			if a.MLPBias != e.MLPBias {
				t.Errorf("mlp_bias = %v, want %v", a.MLPBias, e.MLPBias)
			}
			if a.RopeTraditional != e.RopeTraditional {
				t.Errorf("rope_traditional = %v, want %v", a.RopeTraditional, e.RopeTraditional)
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

func TestLlamaWeightNamesParity(t *testing.T) {
	f := loadLlamaFixture(t)
	for _, c := range f.WeightNames {
		t.Run(c.Label, func(t *testing.T) {
			a, err := ParseLlamaArgs(c.Config)
			if err != nil {
				t.Fatalf("ParseLlamaArgs: %v", err)
			}
			got := a.WeightNames()
			if !reflect.DeepEqual(got, c.Names) {
				t.Errorf("weight names mismatch\n got %v\nwant %v", got, c.Names)
			}
		})
	}
}

func TestLlamaMakeCacheParity(t *testing.T) {
	f := loadLlamaFixture(t)
	for _, c := range f.Cache {
		t.Run(c.Label, func(t *testing.T) {
			a, err := ParseLlamaArgs(c.Config)
			if err != nil {
				t.Fatalf("ParseLlamaArgs: %v", err)
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

func TestParseLlamaArgsDefaults(t *testing.T) {
	// head_dim, num_key_value_heads, rope_theta, and tie_word_embeddings all
	// absent: the reference derives head_dim and kv heads, defaults theta to
	// 10000, and ties the embeddings.
	cfg := `{"hidden_size":32,"num_attention_heads":4,"num_hidden_layers":2,"vocab_size":10}`
	a, err := ParseLlamaArgs([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseLlamaArgs: %v", err)
	}
	checkInt(t, "head_dim", a.HeadDim, 8)
	checkInt(t, "num_key_value_heads", a.NumKeyValueHeads, 4)
	checkFloat(t, "rope_theta", a.RopeTheta, 10000)
	if !a.TieWordEmbeddings {
		t.Error("tie_word_embeddings should default to true")
	}
	if a.AttentionBias || a.MLPBias || a.RopeTraditional {
		t.Error("bias and rope_traditional flags should default to false")
	}
}

func TestParseLlamaArgsErrors(t *testing.T) {
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
			if _, err := ParseLlamaArgs([]byte(c.cfg)); err == nil {
				t.Error("expected error")
			}
		})
	}
}

func TestLlamaSanitizeDropsRotaryAndTiedHead(t *testing.T) {
	arr := func() *mlxgo.Array {
		a, _ := mlxgo.NewFloat32([]float32{0}, 1)
		return a
	}
	tied := &LlamaArgs{TieWordEmbeddings: true}
	w := map[string]*mlxgo.Array{
		"lm_head.weight": arr(),
		"model.layers.0.self_attn.rotary_emb.inv_freq": arr(),
		"model.norm.weight":                            arr(),
	}
	tied.Sanitize(w)
	if _, ok := w["lm_head.weight"]; ok {
		t.Error("tied Sanitize should drop lm_head.weight")
	}
	if _, ok := w["model.layers.0.self_attn.rotary_emb.inv_freq"]; ok {
		t.Error("Sanitize should drop rotary_emb.inv_freq buffers")
	}
	if _, ok := w["model.norm.weight"]; !ok {
		t.Error("Sanitize dropped an unrelated weight")
	}

	untied := &LlamaArgs{TieWordEmbeddings: false}
	w2 := map[string]*mlxgo.Array{
		"lm_head.weight": arr(),
		"model.layers.0.self_attn.rotary_emb.inv_freq": arr(),
	}
	untied.Sanitize(w2)
	if _, ok := w2["lm_head.weight"]; !ok {
		t.Error("untied Sanitize must keep lm_head.weight")
	}
	if _, ok := w2["model.layers.0.self_attn.rotary_emb.inv_freq"]; ok {
		t.Error("Sanitize should drop rotary buffers regardless of tying")
	}
}

func BenchmarkParseLlamaArgs(b *testing.B) {
	b.ReportAllocs()
	cfg := []byte(`{"model_type":"llama","hidden_size":1024,"num_hidden_layers":28,"intermediate_size":3072,"num_attention_heads":16,"rms_norm_eps":1e-5,"vocab_size":128256,"num_key_value_heads":8,"max_position_embeddings":8192,"rope_theta":500000.0,"head_dim":64,"tie_word_embeddings":false}`)
	for b.Loop() {
		a, err := ParseLlamaArgs(cfg)
		if err != nil {
			b.Fatal(err)
		}
		_ = a.WeightNames()
	}
}
