// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"errors"
	"maps"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// tinyGemma4Args is a 4-layer config small enough to assemble dummy weights for.
// The default sliding/full pattern with sliding_window_pattern 2 gives
// [sliding, full, sliding, full]; num_kv_shared_layers 2 makes layers 0 and 1 own
// their caches and layers 2 and 3 share them. hp toggles the per-layer-input path.
func tinyGemma4Args(t *testing.T, tie bool, hp int) *Gemma4TextArgs {
	t.Helper()
	cfg := `{"model_type":"gemma4_text","hidden_size":8,"num_hidden_layers":4,` +
		`"intermediate_size":16,"num_attention_heads":2,"head_dim":4,"global_head_dim":8,` +
		`"rms_norm_eps":1e-6,"vocab_size":32,"num_key_value_heads":1,"num_kv_shared_layers":2,` +
		`"hidden_size_per_layer_input":` + strconv.Itoa(hp) + `,"sliding_window":3,` +
		`"sliding_window_pattern":2,"max_position_embeddings":128,"final_logit_softcapping":30.0,` +
		`"tie_word_embeddings":` + boolStr(tie) + `}`
	a, err := ParseGemma4TextArgs([]byte(cfg))
	if err != nil {
		t.Fatalf("ParseGemma4TextArgs: %v", err)
	}
	return a
}

// dummyGemma4Weights builds a stub array for every weight name the model expects.
// Shapes are irrelevant to assembly; the stub holds them as opaque host data.
func dummyGemma4Weights(t *testing.T, a *Gemma4TextArgs) map[string]*mlxgo.Array {
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

func TestProportionalFreqs(t *testing.T) {
	// Partial rotation of 0.25 over an 8-wide head rotates only the leading 2
	// dimensions: one finite frequency (base^0 = 1) then an +Inf tail.
	got := proportionalFreqs(8, 0.25, 1e6)
	if len(got) != 4 {
		t.Fatalf("len = %d, want 4", len(got))
	}
	if got[0] != 1 {
		t.Errorf("freq[0] = %v, want 1", got[0])
	}
	for i := 1; i < 4; i++ {
		if !math.IsInf(float64(got[i]), 1) {
			t.Errorf("freq[%d] = %v, want +Inf", i, got[i])
		}
	}

	// Full rotation (factor 1.0) leaves no +Inf tail: every frequency is finite
	// and equals base^(2k/head_dim).
	full := proportionalFreqs(8, 1.0, 1e6)
	for k, f := range full {
		want := float32(math.Pow(1e6, float64(2*k)/8))
		if math.IsInf(float64(f), 0) || math.Abs(float64(f-want)) > 1e-3*float64(want) {
			t.Errorf("full freq[%d] = %v, want ~%v", k, f, want)
		}
	}
}

func TestSlidingWindowMask(t *testing.T) {
	// qLen 3, no offset, window 2: each query sees itself, the one before it, and
	// nothing causally ahead or older than the window.
	m, err := slidingWindowMask(3, 0, 2)
	if err != nil {
		t.Fatalf("slidingWindowMask: %v", err)
	}
	if want := []int{1, 1, 3, 3}; !reflect.DeepEqual(m.Shape(), want) {
		t.Fatalf("shape = %v, want %v", m.Shape(), want)
	}
	data, err := m.ToFloat32()
	if err != nil {
		t.Fatalf("ToFloat32: %v", err)
	}
	// row-major [qLen=3, total=3]; 0 means attend, -Inf means masked.
	allow := func(i, j int) bool { return data[i*3+j] == 0 }
	want := [3][3]bool{
		{true, false, false}, // p=0: only key 0
		{true, true, false},  // p=1: keys 0,1
		{false, true, true},  // p=2: keys 1,2 (key 0 falls outside the window)
	}
	for i := range 3 {
		for j := range 3 {
			if allow(i, j) != want[i][j] {
				t.Errorf("mask[%d,%d] allow=%v, want %v", i, j, allow(i, j), want[i][j])
			}
		}
	}
}

func TestSlidingWindowMaskWithOffset(t *testing.T) {
	// One new query at absolute position 5 with a 4-long history and window 3 sees
	// keys 3,4,5 only.
	m, err := slidingWindowMask(1, 5, 3)
	if err != nil {
		t.Fatalf("slidingWindowMask: %v", err)
	}
	data, _ := m.ToFloat32()
	if len(data) != 6 {
		t.Fatalf("len = %d, want 6", len(data))
	}
	for j := range 6 {
		want := j >= 3 && j <= 5
		if (data[j] == 0) != want {
			t.Errorf("key %d allow=%v, want %v", j, data[j] == 0, want)
		}
	}
}

func TestBatchLeftPadSlidingWindowMask(t *testing.T) {
	// Two rows of a ragged cohort: row 0 has no padding, row 1 was left-padded by 2.
	// qLen 2, offset 3, window 2. Row 0 must reproduce slidingWindowMask byte for
	// byte; row 1 must also drop the two padding keys (j < 2) on top of the window.
	leftPad := []int{0, 2}
	const qLen, offset, window = 2, 3, 2
	m, err := batchLeftPadSlidingWindowMask(leftPad, qLen, offset, window)
	if err != nil {
		t.Fatalf("batchLeftPadSlidingWindowMask: %v", err)
	}
	total := offset + qLen
	if want := []int{2, 1, qLen, total}; !reflect.DeepEqual(m.Shape(), want) {
		t.Fatalf("shape = %v, want %v", m.Shape(), want)
	}
	data, err := m.ToFloat32()
	if err != nil {
		t.Fatalf("ToFloat32: %v", err)
	}
	allow := func(b, i, j int) bool { return data[(b*qLen+i)*total+j] == 0 }
	for b, pad := range leftPad {
		for i := range qLen {
			p := offset + i
			for j := range total {
				want := j <= p && p-j < window && j >= pad
				if allow(b, i, j) != want {
					t.Errorf("row %d mask[%d,%d] allow=%v, want %v", b, i, j, allow(b, i, j), want)
				}
			}
		}
	}
	// The zero-padding row must equal the single-sequence builder exactly.
	single, err := slidingWindowMask(qLen, offset, window)
	if err != nil {
		t.Fatalf("slidingWindowMask: %v", err)
	}
	sdata, _ := single.ToFloat32()
	for k := range sdata {
		if data[k] != sdata[k] && !(math.IsInf(float64(data[k]), -1) && math.IsInf(float64(sdata[k]), -1)) {
			t.Errorf("row 0 elem %d = %g, want %g", k, data[k], sdata[k])
		}
	}
}

func TestNewGemma4TextModelWiresWeights(t *testing.T) {
	for _, tie := range []bool{true, false} {
		a := tinyGemma4Args(t, tie, 4)
		m, err := NewGemma4TextModel(a, dummyGemma4Weights(t, a))
		if err != nil {
			t.Fatalf("NewGemma4TextModel(tie=%v): %v", tie, err)
		}
		if len(m.layers) != a.NumLayers() {
			t.Errorf("layers = %d, want %d", len(m.layers), a.NumLayers())
		}
		if m.embedTokens == nil || m.norm == nil {
			t.Error("embedTokens / norm not wired")
		}
		if m.fullFreqs == nil {
			t.Error("fullFreqs not built (config has full-attention layers)")
		}
		if m.embedTokensPerLayer == nil || m.perLayerModelProjection == nil || m.perLayerProjectionNorm == nil {
			t.Error("per-layer-input tensors not wired")
		}
		for i := range m.layers {
			l := &m.layers[i]
			if l.qProj == nil || l.oProj == nil || l.qNorm == nil || l.gateProj == nil ||
				l.upProj == nil || l.downProj == nil || l.layerScalar == nil ||
				l.inputLayernorm == nil || l.preFeedforwardLayernorm == nil {
				t.Errorf("layer %d has an unwired required weight", i)
			}
			if l.perLayerInputGate == nil || l.perLayerProjection == nil || l.postPerLayerInputNorm == nil {
				t.Errorf("layer %d missing per-layer-input gating weight", i)
			}
			// Owning layers (0,1) carry k_proj/v_proj/k_norm; shared layers (2,3) do not.
			if a.HasKV(i) {
				if l.kProj == nil || l.vProj == nil || l.kNorm == nil {
					t.Errorf("owning layer %d missing kv weights", i)
				}
			} else if l.kProj != nil || l.vProj != nil || l.kNorm != nil {
				t.Errorf("shared layer %d should not own kv weights", i)
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

func TestNewGemma4TextModelNoPerLayerInputs(t *testing.T) {
	a := tinyGemma4Args(t, true, 0)
	if a.HasPerLayerInputs() {
		t.Fatal("expected per-layer inputs to be off")
	}
	m, err := NewGemma4TextModel(a, dummyGemma4Weights(t, a))
	if err != nil {
		t.Fatalf("NewGemma4TextModel: %v", err)
	}
	if m.embedTokensPerLayer != nil {
		t.Error("per-layer embedding should be nil when the path is off")
	}
	if m.layers[0].perLayerInputGate != nil {
		t.Error("per-layer gate should be nil when the path is off")
	}
}

func TestNewGemma4TextModelRejectsMoE(t *testing.T) {
	a := tinyGemma4Args(t, true, 4)
	a.EnableMoEBlock = true
	if _, err := NewGemma4TextModel(a, dummyGemma4Weights(t, a)); err == nil ||
		!strings.Contains(err.Error(), "enable_moe_block") {
		t.Errorf("err = %v, want a MoE rejection", err)
	}
}

func TestNewGemma4TextModelRejectsAllShared(t *testing.T) {
	a := tinyGemma4Args(t, true, 4)
	a.NumKVSharedLayers = a.NumLayers() // no owning layers
	if _, err := NewGemma4TextModel(a, dummyGemma4Weights(t, a)); err == nil {
		t.Error("expected an error when every layer is KV-shared")
	}
}

func TestNewGemma4TextModelMissingWeight(t *testing.T) {
	a := tinyGemma4Args(t, false, 4)
	w := dummyGemma4Weights(t, a)
	delete(w, "model.layers.1.mlp.down_proj.weight")
	if _, err := NewGemma4TextModel(a, w); err == nil {
		t.Error("expected an error for a missing weight")
	}

	w2 := dummyGemma4Weights(t, a)
	delete(w2, "lm_head.weight")
	if _, err := NewGemma4TextModel(a, w2); err == nil {
		t.Error("expected an error for a missing lm_head")
	}
}

func TestGemma4ForwardGracefulWithoutBackend(t *testing.T) {
	// The forward type-checks and runs against the stub up to the first kernel
	// (the embedding take), then returns ErrMLXUnavailable instead of panicking.
	a := tinyGemma4Args(t, true, 4)
	m, err := NewGemma4TextModel(a, dummyGemma4Weights(t, a))
	if err != nil {
		t.Fatalf("NewGemma4TextModel: %v", err)
	}
	caches := make([]*KVTensorCache, a.NumLayers())
	for i := range caches {
		caches[i] = &KVTensorCache{}
	}
	if _, err := m.Forward([]int32{1, 2, 3}, caches, mlxgo.DefaultStream()); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Errorf("Forward err = %v, want ErrMLXUnavailable", err)
	}
}

func TestGemma4ForwardCacheCountMismatch(t *testing.T) {
	a := tinyGemma4Args(t, true, 4)
	m, err := NewGemma4TextModel(a, dummyGemma4Weights(t, a))
	if err != nil {
		t.Fatalf("NewGemma4TextModel: %v", err)
	}
	if _, err := m.Forward([]int32{1}, nil, mlxgo.DefaultStream()); err == nil {
		t.Error("expected a cache-count mismatch error")
	}
}

func BenchmarkNewGemma4TextModel(b *testing.B) {
	b.ReportAllocs()
	cfg := `{"model_type":"gemma4_text","hidden_size":8,"num_hidden_layers":4,` +
		`"intermediate_size":16,"num_attention_heads":2,"head_dim":4,"global_head_dim":8,` +
		`"rms_norm_eps":1e-6,"vocab_size":32,"num_key_value_heads":1,"num_kv_shared_layers":2,` +
		`"hidden_size_per_layer_input":4,"sliding_window":3,"sliding_window_pattern":2,` +
		`"max_position_embeddings":128,"final_logit_softcapping":30.0,"tie_word_embeddings":true}`
	a, err := ParseGemma4TextArgs([]byte(cfg))
	if err != nil {
		b.Fatalf("ParseGemma4TextArgs: %v", err)
	}
	w := make(map[string]*mlxgo.Array)
	for _, name := range a.WeightNames() {
		arr, _ := mlxgo.NewFloat32([]float32{0}, 1)
		w[name] = arr
	}
	for b.Loop() {
		cp := make(map[string]*mlxgo.Array, len(w))
		maps.Copy(cp, w)
		if _, err := NewGemma4TextModel(a, cp); err != nil {
			b.Fatalf("NewGemma4TextModel: %v", err)
		}
	}
}
