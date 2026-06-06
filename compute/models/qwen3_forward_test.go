// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"errors"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// tinyQwen3Args is a 2-layer config small enough to assemble dummy weights for.
func tinyQwen3Args(t *testing.T, tie bool) *Qwen3Args {
	t.Helper()
	cfg := `{"model_type":"qwen3","hidden_size":8,"num_hidden_layers":2,"intermediate_size":16,` +
		`"num_attention_heads":4,"rms_norm_eps":1e-6,"vocab_size":32,"num_key_value_heads":2,` +
		`"max_position_embeddings":128,"rope_theta":1000000.0,"head_dim":2,"tie_word_embeddings":` +
		boolStr(tie) + `}`
	a, err := ParseQwen3Args([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseQwen3Args: %v", err)
	}
	return a
}

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

// dummyWeights builds a stub array for every weight name the model expects. The
// shapes do not need to be correct for the assembly test; the stub holds them
// as opaque host data.
func dummyWeights(t *testing.T, a *Qwen3Args) map[string]*mlxgo.Array {
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

func TestNewQwen3ModelWiresWeights(t *testing.T) {
	for _, tie := range []bool{true, false} {
		a := tinyQwen3Args(t, tie)
		w := dummyWeights(t, a)
		m, err := NewQwen3Model(a, w)
		if err != nil {
			t.Fatalf("NewQwen3Model(tie=%v): %v", tie, err)
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
				l.qNorm == nil || l.kNorm == nil || l.gateProj == nil || l.upProj == nil ||
				l.downProj == nil || l.inputLayernorm == nil || l.postAttentionLayernorm == nil {
				t.Errorf("layer %d has an unwired weight", i)
			}
		}
		// Tied: no separate head, reuses the embedding table. Untied: real head.
		if tie && m.lmHead != nil {
			t.Error("tied model should have a nil lmHead")
		}
		if !tie && m.lmHead == nil {
			t.Error("untied model should wire lmHead")
		}
	}
}

// tinyQwen3QuantArgs is tinyQwen3Args with a top-level affine quantization block,
// the config a packed checkpoint ships.
func tinyQwen3QuantArgs(t *testing.T, tie bool) *Qwen3Args {
	t.Helper()
	cfg := `{"model_type":"qwen3","hidden_size":8,"num_hidden_layers":2,"intermediate_size":16,` +
		`"num_attention_heads":4,"rms_norm_eps":1e-6,"vocab_size":32,"num_key_value_heads":2,` +
		`"max_position_embeddings":128,"rope_theta":1000000.0,"head_dim":2,"tie_word_embeddings":` +
		boolStr(tie) + `,"quantization":{"group_size":64,"bits":4,"mode":"affine"}}`
	a, err := ParseQwen3Args([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseQwen3Args: %v", err)
	}
	return a
}

// quantizableModule reports whether a "*.weight" key names a module that the
// reference packs (the projections, the embedding, and the untied head), as
// opposed to a never-quantized norm weight.
func quantizableModule(name string) bool {
	for _, suffix := range []string{
		"embed_tokens.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight",
		"o_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight",
		"gate_up_proj.weight", "qkv_proj.weight", "lm_head.weight",
	} {
		if len(name) >= len(suffix) && name[len(name)-len(suffix):] == suffix {
			return true
		}
	}
	return false
}

// dummyQuantWeights builds the dense weights plus a scales/biases sibling for
// every quantizable module, the key layout a packed checkpoint loads from.
func dummyQuantWeights(t *testing.T, a *Qwen3Args) map[string]*mlxgo.Array {
	t.Helper()
	w := dummyWeights(t, a)
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

func TestParseQwen3ArgsCapturesQuantization(t *testing.T) {
	a := tinyQwen3QuantArgs(t, true)
	if a.quant != (quantConfig{GroupSize: 64, Bits: 4}) {
		t.Errorf("quant = %+v, want {64 4}", a.quant)
	}
	// The default unquantized config leaves the geometry zero.
	if d := tinyQwen3Args(t, true).quant; d.quantized() {
		t.Errorf("dense config reported quantized: %+v", d)
	}
}

func TestNewQwen3ModelQuantizedWiring(t *testing.T) {
	a := tinyQwen3QuantArgs(t, false)
	m, err := NewQwen3Model(a, dummyQuantWeights(t, a))
	if err != nil {
		t.Fatalf("NewQwen3Model: %v", err)
	}
	if !m.embedTokens.isQuantized() || !m.lmHead.isQuantized() {
		t.Error("embedding / head not loaded quantized")
	}
	for i := range m.layers {
		l := &m.layers[i]
		for _, q := range []*qLinear{l.qProj, l.kProj, l.vProj, l.oProj, l.gateProj, l.upProj, l.downProj} {
			if !q.isQuantized() {
				t.Fatalf("layer %d projection not loaded quantized", i)
			}
			if q.groupSize != 64 || q.bits != 4 {
				t.Fatalf("layer %d geometry = gs%d/b%d, want 64/4", i, q.groupSize, q.bits)
			}
		}
	}
}

func TestNewQwen3ModelQuantizedMissingBiases(t *testing.T) {
	a := tinyQwen3QuantArgs(t, true)
	w := dummyQuantWeights(t, a)
	delete(w, "model.layers.0.self_attn.q_proj.biases")
	if _, err := NewQwen3Model(a, w); err == nil {
		t.Error("expected an error for a quantized module missing its biases")
	}
}

func TestQwen3QuantizedForwardReachesSeam(t *testing.T) {
	// A quantized model type-checks and routes through the quantized kernels,
	// reaching ErrMLXUnavailable at the first one under the stub.
	a := tinyQwen3QuantArgs(t, true)
	m, err := NewQwen3Model(a, dummyQuantWeights(t, a))
	if err != nil {
		t.Fatalf("NewQwen3Model: %v", err)
	}
	caches := make([]*KVTensorCache, a.NumHiddenLayers)
	for i := range caches {
		caches[i] = &KVTensorCache{}
	}
	if _, err := m.Forward([]int32{1, 2, 3}, caches, mlxgo.DefaultStream()); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Errorf("Forward err = %v, want ErrMLXUnavailable", err)
	}
}

func TestNewQwen3ModelMissingWeight(t *testing.T) {
	a := tinyQwen3Args(t, false)
	w := dummyWeights(t, a)
	delete(w, "model.layers.1.mlp.down_proj.weight")
	if _, err := NewQwen3Model(a, w); err == nil {
		t.Error("expected an error for a missing weight")
	}

	// Untied model with the head missing.
	w2 := dummyWeights(t, a)
	delete(w2, "lm_head.weight")
	if _, err := NewQwen3Model(a, w2); err == nil {
		t.Error("expected an error for a missing lm_head")
	}
}

func TestKVTensorCacheFirstUpdateStores(t *testing.T) {
	// First update stores its inputs without a concatenate, so it works in the
	// stub and the offset reflects the stored sequence length (axis 2).
	c := &KVTensorCache{}
	k, err := mlxgo.NewFloat32(make([]float32, 1*2*3*2), 1, 2, 3, 2)
	if err != nil {
		t.Fatalf("NewFloat32 keys: %v", err)
	}
	v, err := mlxgo.NewFloat32(make([]float32, 1*2*3*2), 1, 2, 3, 2)
	if err != nil {
		t.Fatalf("NewFloat32 values: %v", err)
	}
	gotK, gotV, err := c.Update(k, v, mlxgo.DefaultStream())
	if err != nil {
		t.Fatalf("first Update: %v", err)
	}
	if gotK != k || gotV != v {
		t.Error("first Update should return the stored tensors unchanged")
	}
	if c.Offset != 3 {
		t.Errorf("Offset = %d, want 3", c.Offset)
	}
	if c.Keys() != k || c.Values() != v {
		t.Error("Keys/Values accessors out of sync")
	}
}

func TestKVTensorCacheSecondUpdateNeedsBackend(t *testing.T) {
	// A second update concatenates, which needs the real backend; under the stub
	// it surfaces ErrMLXUnavailable rather than corrupting the cache.
	c := &KVTensorCache{}
	mk := func() *mlxgo.Array {
		arr, _ := mlxgo.NewFloat32(make([]float32, 1*2*1*2), 1, 2, 1, 2)
		return arr
	}
	if _, _, err := c.Update(mk(), mk(), mlxgo.DefaultStream()); err != nil {
		t.Fatalf("first Update: %v", err)
	}
	if _, _, err := c.Update(mk(), mk(), mlxgo.DefaultStream()); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Errorf("second Update err = %v, want ErrMLXUnavailable", err)
	}
}

func TestQwen3ForwardGracefulWithoutBackend(t *testing.T) {
	// The forward type-checks and runs against the stub up to the first kernel
	// (the embedding take), then returns ErrMLXUnavailable instead of panicking.
	a := tinyQwen3Args(t, true)
	m, err := NewQwen3Model(a, dummyWeights(t, a))
	if err != nil {
		t.Fatalf("NewQwen3Model: %v", err)
	}
	caches := make([]*KVTensorCache, a.NumHiddenLayers)
	for i := range caches {
		caches[i] = &KVTensorCache{}
	}
	if _, err := m.Forward([]int32{1, 2, 3}, caches, mlxgo.DefaultStream()); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Errorf("Forward err = %v, want ErrMLXUnavailable", err)
	}
}

func TestQwen3ForwardCacheCountMismatch(t *testing.T) {
	a := tinyQwen3Args(t, true)
	m, err := NewQwen3Model(a, dummyWeights(t, a))
	if err != nil {
		t.Fatalf("NewQwen3Model: %v", err)
	}
	if _, err := m.Forward([]int32{1}, nil, mlxgo.DefaultStream()); err == nil {
		t.Error("expected a cache-count mismatch error")
	}
}
