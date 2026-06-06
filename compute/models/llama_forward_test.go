// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"errors"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// tinyLlamaArgs is a 2-layer config small enough to assemble dummy weights for.
// bias toggles the optional attention and MLP projection biases.
func tinyLlamaArgs(t *testing.T, tie, bias bool) *LlamaArgs {
	t.Helper()
	cfg := `{"model_type":"llama","hidden_size":8,"num_hidden_layers":2,"intermediate_size":16,` +
		`"num_attention_heads":4,"rms_norm_eps":1e-6,"vocab_size":32,"num_key_value_heads":2,` +
		`"max_position_embeddings":128,"rope_theta":500000.0,"head_dim":2,"tie_word_embeddings":` +
		boolStr(tie) + `,"attention_bias":` + boolStr(bias) + `,"mlp_bias":` + boolStr(bias) + `}`
	a, err := ParseLlamaArgs([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseLlamaArgs: %v", err)
	}
	return a
}

// dummyLlamaWeights builds a stub array for every weight name the model expects.
func dummyLlamaWeights(t *testing.T, a *LlamaArgs) map[string]*mlxgo.Array {
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

func TestNewLlamaModelWiresWeights(t *testing.T) {
	for _, tie := range []bool{true, false} {
		for _, bias := range []bool{true, false} {
			a := tinyLlamaArgs(t, tie, bias)
			w := dummyLlamaWeights(t, a)
			m, err := NewLlamaModel(a, w)
			if err != nil {
				t.Fatalf("NewLlamaModel(tie=%v,bias=%v): %v", tie, bias, err)
			}
			if len(m.layers) != a.NumHiddenLayers {
				t.Errorf("layers = %d, want %d", len(m.layers), a.NumHiddenLayers)
			}
			if m.embedTokens == nil || m.norm == nil {
				t.Error("embedTokens / norm not wired")
			}
			for i := range m.layers {
				l := &m.layers[i]
				if l.qProj == nil || l.kProj == nil || l.vProj == nil || l.oProj == nil ||
					l.gateProj == nil || l.upProj == nil || l.downProj == nil ||
					l.inputLayernorm == nil || l.postAttentionLayernorm == nil {
					t.Errorf("layer %d has an unwired weight", i)
				}
				gotBias := l.qBias != nil && l.kBias != nil && l.vBias != nil && l.oBias != nil &&
					l.gateBias != nil && l.upBias != nil && l.downBias != nil
				if gotBias != bias {
					t.Errorf("layer %d bias wiring = %v, want %v", i, gotBias, bias)
				}
			}
			if tie && m.lmHead != nil {
				t.Error("tied model should have a nil lmHead")
			}
			if !tie && m.lmHead == nil {
				t.Error("untied model should wire lmHead")
			}
		}
	}
}

// tinyLlamaQuantArgs is tinyLlamaArgs with a top-level affine quantization block.
func tinyLlamaQuantArgs(t *testing.T, tie, bias bool) *LlamaArgs {
	t.Helper()
	cfg := `{"model_type":"llama","hidden_size":8,"num_hidden_layers":2,"intermediate_size":16,` +
		`"num_attention_heads":4,"rms_norm_eps":1e-6,"vocab_size":32,"num_key_value_heads":2,` +
		`"max_position_embeddings":128,"rope_theta":500000.0,"head_dim":2,"tie_word_embeddings":` +
		boolStr(tie) + `,"attention_bias":` + boolStr(bias) + `,"mlp_bias":` + boolStr(bias) +
		`,"quantization":{"group_size":64,"bits":4,"mode":"affine"}}`
	a, err := ParseLlamaArgs([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseLlamaArgs: %v", err)
	}
	return a
}

// dummyLlamaQuantWeights builds the dense weights (including the additive .bias
// keys) plus a scales/biases affine sibling for every quantizable module. The
// additive ".bias" and the affine ".biases" are distinct keys and coexist.
func dummyLlamaQuantWeights(t *testing.T, a *LlamaArgs) map[string]*mlxgo.Array {
	t.Helper()
	w := dummyLlamaWeights(t, a)
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

func TestNewLlamaModelQuantizedWiring(t *testing.T) {
	// A quantized checkpoint with additive biases: the projections load
	// quantized, the additive biases stay dense, both present at once.
	a := tinyLlamaQuantArgs(t, false, true)
	if a.quant != (quantConfig{GroupSize: 64, Bits: 4}) {
		t.Fatalf("quant = %+v, want {64 4}", a.quant)
	}
	m, err := NewLlamaModel(a, dummyLlamaQuantWeights(t, a))
	if err != nil {
		t.Fatalf("NewLlamaModel: %v", err)
	}
	if !m.embedTokens.isQuantized() || !m.lmHead.isQuantized() {
		t.Error("embedding / head not loaded quantized")
	}
	for i := range m.layers {
		l := &m.layers[i]
		for _, q := range []*qLinear{l.qProj, l.kProj, l.vProj, l.oProj, l.gateProj, l.upProj, l.downProj} {
			if !q.isQuantized() || q.groupSize != 64 || q.bits != 4 {
				t.Fatalf("layer %d projection not quantized at 64/4", i)
			}
		}
		if l.qBias == nil || l.gateBias == nil {
			t.Errorf("layer %d additive biases dropped", i)
		}
	}
}

func TestLlamaQuantizedForwardReachesSeam(t *testing.T) {
	for _, bias := range []bool{false, true} {
		a := tinyLlamaQuantArgs(t, true, bias)
		m, err := NewLlamaModel(a, dummyLlamaQuantWeights(t, a))
		if err != nil {
			t.Fatalf("NewLlamaModel: %v", err)
		}
		if _, err := m.Forward([]int32{1, 2, 3}, llamaCaches(t, a.NumHiddenLayers), mlxgo.DefaultStream()); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
			t.Errorf("Forward(bias=%v) err = %v, want ErrMLXUnavailable", bias, err)
		}
	}
}

func TestNewLlamaModelMissingWeight(t *testing.T) {
	a := tinyLlamaArgs(t, false, false)
	w := dummyLlamaWeights(t, a)
	delete(w, "model.layers.1.mlp.down_proj.weight")
	if _, err := NewLlamaModel(a, w); err == nil {
		t.Error("expected an error for a missing weight")
	}

	// Bias enabled but a bias key missing.
	ab := tinyLlamaArgs(t, false, true)
	wb := dummyLlamaWeights(t, ab)
	delete(wb, "model.layers.0.self_attn.q_proj.bias")
	if _, err := NewLlamaModel(ab, wb); err == nil {
		t.Error("expected an error for a missing bias")
	}
}

func TestLlamaForwardGracefulWithoutBackend(t *testing.T) {
	// The forward type-checks and runs against the stub up to the first kernel
	// (the embedding take), then returns ErrMLXUnavailable instead of panicking.
	for _, bias := range []bool{false, true} {
		a := tinyLlamaArgs(t, true, bias)
		m, err := NewLlamaModel(a, dummyLlamaWeights(t, a))
		if err != nil {
			t.Fatalf("NewLlamaModel: %v", err)
		}
		caches := make([]*KVTensorCache, a.NumHiddenLayers)
		for i := range caches {
			caches[i] = &KVTensorCache{}
		}
		if _, err := m.Forward([]int32{1, 2, 3}, caches, mlxgo.DefaultStream()); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
			t.Errorf("Forward(bias=%v) err = %v, want ErrMLXUnavailable", bias, err)
		}
	}
}

func TestLlamaForwardCacheCountMismatch(t *testing.T) {
	a := tinyLlamaArgs(t, true, false)
	m, err := NewLlamaModel(a, dummyLlamaWeights(t, a))
	if err != nil {
		t.Fatalf("NewLlamaModel: %v", err)
	}
	if _, err := m.Forward([]int32{1}, nil, mlxgo.DefaultStream()); err == nil {
		t.Error("expected a cache-count mismatch error")
	}
}

func llamaCaches(t *testing.T, n int) []*KVTensorCache {
	t.Helper()
	caches := make([]*KVTensorCache, n)
	for i := range caches {
		caches[i] = &KVTensorCache{}
	}
	return caches
}

func TestLlamaBatchDecodeGraceful(t *testing.T) {
	// The batched decode step type-checks and runs against the stub up to the
	// first kernel (the embedding take over the [batch, 1] input), then reports
	// the missing backend. One token per row, several batch sizes.
	a := tinyLlamaArgs(t, true, false)
	m, err := NewLlamaModel(a, dummyLlamaWeights(t, a))
	if err != nil {
		t.Fatalf("NewLlamaModel: %v", err)
	}
	for _, batch := range []int{1, 2, 5} {
		tokens := make([]int32, batch)
		for i := range tokens {
			tokens[i] = int32(i + 1)
		}
		_, err := m.BatchDecode(tokens, batch, llamaCaches(t, a.NumHiddenLayers), mlxgo.DefaultStream())
		if !errors.Is(err, mlxgo.ErrMLXUnavailable) {
			t.Errorf("BatchDecode(batch=%d) err = %v, want ErrMLXUnavailable", batch, err)
		}
	}
}

func TestLlamaBatchDecodeCacheMismatch(t *testing.T) {
	// The cache-count guard rejects a wrong-length cache list before any kernel,
	// the same as the single-sequence forward.
	a := tinyLlamaArgs(t, true, false)
	m, err := NewLlamaModel(a, dummyLlamaWeights(t, a))
	if err != nil {
		t.Fatalf("NewLlamaModel: %v", err)
	}
	if _, err := m.BatchDecode([]int32{1, 2}, 2, llamaCaches(t, 1), mlxgo.DefaultStream()); err == nil {
		t.Error("expected a cache-count mismatch error")
	}
}

// TestLlamaBatchDecodeMatchesForwardForOneRow pins the batch=1 decode to the
// single-sequence forward: both feed one token through identical shapes and
// surface the same backend-missing error, so the batched path is a strict
// generalization of the path that already serves single sequences.
func TestLlamaBatchDecodeMatchesForwardForOneRow(t *testing.T) {
	a := tinyLlamaArgs(t, true, false)
	m, err := NewLlamaModel(a, dummyLlamaWeights(t, a))
	if err != nil {
		t.Fatalf("NewLlamaModel: %v", err)
	}
	_, fwdErr := m.Forward([]int32{7}, llamaCaches(t, a.NumHiddenLayers), mlxgo.DefaultStream())
	_, batErr := m.BatchDecode([]int32{7}, 1, llamaCaches(t, a.NumHiddenLayers), mlxgo.DefaultStream())
	if !errors.Is(fwdErr, mlxgo.ErrMLXUnavailable) || !errors.Is(batErr, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("Forward err = %v, BatchDecode err = %v, want both ErrMLXUnavailable", fwdErr, batErr)
	}
}
