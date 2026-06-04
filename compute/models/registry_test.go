// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// minimalQwen3Config is a small but valid config.json body the qwen3 builder
// accepts. Two layers keep the cache assertion meaningful and the fabricated
// weight map small.
const minimalQwen3Config = `{
	"model_type": "qwen3",
	"hidden_size": 8,
	"num_hidden_layers": 2,
	"intermediate_size": 16,
	"num_attention_heads": 4,
	"num_key_value_heads": 2,
	"head_dim": 2,
	"vocab_size": 32,
	"rms_norm_eps": 1e-6,
	"rope_theta": 10000.0,
	"tie_word_embeddings": false
}`

// fabricateWeights returns a zero placeholder array for every name, so the
// concrete model wires up on the host (construction stores the arrays; only the
// forward needs real values and the GPU).
func fabricateWeights(t *testing.T, names []string) map[string]*mlxgo.Array {
	t.Helper()
	w := make(map[string]*mlxgo.Array, len(names))
	for _, n := range names {
		a, err := mlxgo.Zeros(mlxgo.Float32, 1)
		if err != nil {
			t.Fatalf("Zeros: %v", err)
		}
		w[n] = a
	}
	return w
}

func TestBuildModelQwen3(t *testing.T) {
	args, err := ParseQwen3Args([]byte(minimalQwen3Config))
	if err != nil {
		t.Fatalf("ParseQwen3Args: %v", err)
	}
	weights := fabricateWeights(t, args.WeightNames())

	m, err := BuildModel("qwen3", []byte(minimalQwen3Config), weights, 7)
	if err != nil {
		t.Fatalf("BuildModel: %v", err)
	}
	if got := m.EOS(); got != 7 {
		t.Fatalf("EOS = %d, want 7", got)
	}
	caches, ok := m.NewCache().([]*KVTensorCache)
	if !ok {
		t.Fatalf("NewCache returned %T, want []*KVTensorCache", m.NewCache())
	}
	if len(caches) != 2 {
		t.Fatalf("NewCache made %d caches, want 2 (num_hidden_layers)", len(caches))
	}
	// The forward runs on the host up to the first kernel op, then reports the
	// missing backend; that is the wiring confirmation.
	if _, err := m.Forward([]int32{1}, caches, nil); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("Forward err = %v, want ErrMLXUnavailable", err)
	}
}

func TestBuildModelUnknownType(t *testing.T) {
	_, err := BuildModel("totally_made_up", []byte(`{}`), nil, 0)
	if err == nil {
		t.Fatal("BuildModel accepted an unknown model_type")
	}
	if !strings.Contains(err.Error(), "totally_made_up") {
		t.Fatalf("error %q does not name the unsupported type", err)
	}
}

func TestBuildModelConfigErrorPropagates(t *testing.T) {
	// A known type with an invalid config must surface the parser's error, not
	// build a broken model.
	_, err := BuildModel("qwen3", []byte(`{"model_type":"qwen3"}`), nil, 0)
	if err == nil {
		t.Fatal("BuildModel accepted a config with no positive dims")
	}
}

func TestBuildModelMissingWeight(t *testing.T) {
	// Valid config, empty weight map: the concrete constructor must report the
	// first missing tensor rather than panicking later.
	_, err := BuildModel("qwen3", []byte(minimalQwen3Config), map[string]*mlxgo.Array{}, 0)
	if err == nil {
		t.Fatal("BuildModel accepted an empty weight map")
	}
	if !strings.Contains(err.Error(), "missing weight") {
		t.Fatalf("error %q is not the missing-weight report", err)
	}
}

func TestRegisteredModelTypes(t *testing.T) {
	got := RegisteredModelTypes()
	want := []string{"deepseek_v3", "gemma4_text", "glm4", "llama", "ministral3", "phi3", "qwen3"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("RegisteredModelTypes = %v, want %v", got, want)
	}
}

func BenchmarkBuildModelQwen3(b *testing.B) {
	args, _ := ParseQwen3Args([]byte(minimalQwen3Config))
	names := args.WeightNames()
	weights := make(map[string]*mlxgo.Array, len(names))
	for _, n := range names {
		weights[n], _ = mlxgo.Zeros(mlxgo.Float32, 1)
	}
	cfg := []byte(minimalQwen3Config)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := BuildModel("qwen3", cfg, weights, 7); err != nil {
			b.Fatalf("BuildModel: %v", err)
		}
	}
}
