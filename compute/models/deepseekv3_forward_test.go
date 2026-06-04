// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"errors"
	"math"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// dummyDeepseekWeights builds a stub array for every weight name the model
// expects. The shapes are opaque to the stub; only the key set matters for the
// assembly and graceful-degradation tests.
func dummyDeepseekWeights(t *testing.T, a *DeepseekV3Args) map[string]*mlxgo.Array {
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

func parseDeepseek(t *testing.T, overrides map[string]any) *DeepseekV3Args {
	t.Helper()
	a, err := ParseDeepseekV3Args(baseDeepseekConfig(overrides))
	if err != nil {
		t.Fatalf("ParseDeepseekV3Args: %v", err)
	}
	return a
}

func TestNewDeepseekV3ModelWiresWeights(t *testing.T) {
	for _, qlora := range []bool{true, false} {
		over := map[string]any{}
		if !qlora {
			over["q_lora_rank"] = nil // explicit null selects the plain q_proj path
		}
		a := parseDeepseek(t, over)
		if a.HasQLora() != qlora {
			t.Fatalf("HasQLora = %v, want %v", a.HasQLora(), qlora)
		}
		m, err := NewDeepseekV3Model(a, dummyDeepseekWeights(t, a))
		if err != nil {
			t.Fatalf("NewDeepseekV3Model(qlora=%v): %v", qlora, err)
		}
		if len(m.layers) != a.NumLayers() {
			t.Errorf("layers = %d, want %d", len(m.layers), a.NumLayers())
		}
		if m.embedTokens == nil || m.norm == nil || m.lmHead == nil {
			t.Error("embedTokens / norm / lmHead not wired")
		}
		// Layer 0 is dense (first_k_dense_replace = 1); layers 1..3 are routed.
		dense := &m.layers[0]
		if !(dense.gateProj != nil && dense.upProj != nil && dense.downProj != nil) || dense.isMoE {
			t.Error("layer 0 should be a wired dense MLP")
		}
		moe := &m.layers[1]
		if !moe.isMoE || moe.gateW == nil || moe.eScoreBias == nil ||
			moe.switchGate == nil || moe.switchUp == nil || moe.switchDown == nil ||
			moe.sharedGate == nil || moe.sharedUp == nil || moe.sharedDown == nil {
			t.Error("layer 1 should be a wired routed MLP with a shared expert")
		}
		// Attention path matches the q-lora choice.
		if qlora && (moe.qAProj == nil || moe.qALayernorm == nil || moe.qBProj == nil || moe.qProj != nil) {
			t.Error("q-lora layer should wire q_a/q_b and leave q_proj nil")
		}
		if !qlora && (moe.qProj == nil || moe.qAProj != nil) {
			t.Error("plain layer should wire q_proj and leave q_a nil")
		}
		if moe.embedQ == nil || moe.unembedOut == nil || moe.kvAProj == nil ||
			moe.kvALayernorm == nil || moe.oProj == nil {
			t.Error("latent attention projections not wired")
		}
	}
}

func TestNewDeepseekV3ModelMissingWeight(t *testing.T) {
	a := parseDeepseek(t, nil)
	w := dummyDeepseekWeights(t, a)
	delete(w, "model.layers.1.mlp.switch_mlp.down_proj.weight")
	if _, err := NewDeepseekV3Model(a, w); err == nil {
		t.Error("expected an error for a missing routed weight")
	}

	w2 := dummyDeepseekWeights(t, a)
	delete(w2, "lm_head.weight")
	if _, err := NewDeepseekV3Model(a, w2); err == nil {
		t.Error("expected an error for a missing lm_head")
	}
}

func TestDeepseekV3ModelAttentionBias(t *testing.T) {
	// With attention_bias on, the optional q_a/kv_a/o biases load when present.
	a := parseDeepseek(t, map[string]any{"attention_bias": true})
	w := dummyDeepseekWeights(t, a)
	m, err := NewDeepseekV3Model(a, w)
	if err != nil {
		t.Fatalf("NewDeepseekV3Model: %v", err)
	}
	ly := &m.layers[1]
	if ly.qAProjBias == nil || ly.kvAProjBias == nil || ly.oProjBias == nil {
		t.Error("attention biases should be wired when attention_bias is set")
	}
}

func TestDeepseekV3ForwardGraceful(t *testing.T) {
	// The forward type-checks and runs against the stub up to the first kernel
	// (the embedding take), then reports ErrMLXUnavailable instead of panicking.
	// Both the prefill (L>1) and decode (L==1) token counts are exercised, across
	// the q-lora and plain query paths and the plain and yarn rope schedules.
	cases := []struct {
		name   string
		over   map[string]any
		tokens []int32
	}{
		{"qlora-prefill", nil, []int32{1, 2, 3}},
		{"qlora-decode", nil, []int32{1}},
		{"plain-prefill", map[string]any{"q_lora_rank": nil}, []int32{4, 5}},
		{"plain-decode", map[string]any{"q_lora_rank": nil}, []int32{4}},
		{"yarn-prefill", map[string]any{"rope_scaling": map[string]any{
			"type": "yarn", "factor": 40.0, "mscale_all_dim": 1.0,
			"original_max_position_embeddings": 4096, "beta_fast": 32.0, "beta_slow": 1.0,
		}}, []int32{7, 8, 9}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := parseDeepseek(t, tc.over)
			m, err := NewDeepseekV3Model(a, dummyDeepseekWeights(t, a))
			if err != nil {
				t.Fatalf("NewDeepseekV3Model: %v", err)
			}
			caches := make([]*KVTensorCache, a.NumLayers())
			for i := range caches {
				caches[i] = &KVTensorCache{}
			}
			if _, err := m.Forward(tc.tokens, caches, mlxgo.DefaultStream()); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
				t.Errorf("Forward err = %v, want ErrMLXUnavailable", err)
			}
		})
	}
}

func TestDeepseekV3ForwardCacheMismatch(t *testing.T) {
	a := parseDeepseek(t, nil)
	m, err := NewDeepseekV3Model(a, dummyDeepseekWeights(t, a))
	if err != nil {
		t.Fatalf("NewDeepseekV3Model: %v", err)
	}
	if _, err := m.Forward([]int32{1}, nil, mlxgo.DefaultStream()); err == nil {
		t.Error("expected a cache-count mismatch error")
	}
}

func TestDeepseekV3UnsupportedRope(t *testing.T) {
	a := parseDeepseek(t, map[string]any{"rope_scaling": map[string]any{"type": "mrope", "factor": 2.0}})
	if _, err := NewDeepseekV3Model(a, dummyDeepseekWeights(t, a)); err == nil {
		t.Error("expected an error for an unsupported rope type")
	}
}

// TestRouteExpertsGraceful drives the GPU routing port directly. It runs the
// host shape choreography (the group reshape, the slices) and reports the
// missing backend at the first kernel (the sigmoid) rather than panicking, for
// both the grouped and ungrouped paths.
func TestRouteExpertsGraceful(t *testing.T) {
	for _, nGroup := range []int{4, 1} {
		a := parseDeepseek(t, map[string]any{"n_group": nGroup, "topk_group": min(nGroup, 2)})
		gates, err := mlxgo.NewFloat32(make([]float32, a.NRoutedExperts), 1, 1, a.NRoutedExperts)
		if err != nil {
			t.Fatalf("NewFloat32: %v", err)
		}
		bias, err := mlxgo.NewFloat32(make([]float32, a.NRoutedExperts), a.NRoutedExperts)
		if err != nil {
			t.Fatalf("NewFloat32: %v", err)
		}
		b := &fb{s: mlxgo.DefaultStream()}
		inds, weights := b.routeExperts(gates, bias, a, 1, 1)
		if inds != nil || weights != nil {
			t.Errorf("nGroup=%d: routeExperts = (%v,%v), want (nil,nil) on the stub", nGroup, inds, weights)
		}
		if !errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
			t.Errorf("nGroup=%d: err = %v, want ErrMLXUnavailable", nGroup, b.err)
		}
	}
}

func TestMultiLinearGraceful(t *testing.T) {
	x, _ := mlxgo.NewFloat32(make([]float32, 8), 1, 2, 4)
	w, _ := mlxgo.NewFloat32(make([]float32, 24), 2, 3, 4)
	for _, transpose := range []bool{true, false} {
		b := &fb{s: mlxgo.DefaultStream()}
		if got := b.multiLinear(x, w, transpose); got != nil {
			t.Errorf("transpose=%v: multiLinear = %v, want nil on the stub", transpose, got)
		}
		if !errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
			t.Errorf("transpose=%v: err = %v, want ErrMLXUnavailable", transpose, b.err)
		}
	}
}

// TestDeepseekCausalMask checks the additive prefill mask is built on the host
// (no kernel) with the right shape and the causal triangle: future keys carry a
// large negative bias and at-or-before keys carry zero.
func TestDeepseekCausalMask(t *testing.T) {
	const L, offset = 3, 2
	m := causalAdditiveMask(L, offset)
	S := offset + L
	if len(m) != L*S {
		t.Fatalf("len = %d, want %d", len(m), L*S)
	}
	for i := range L {
		qpos := offset + i
		for j := range S {
			got := m[i*S+j]
			if j <= qpos && got != 0 {
				t.Errorf("mask[%d,%d] = %g, want 0 (attendable)", i, j, got)
			}
			if j > qpos && got >= 0 {
				t.Errorf("mask[%d,%d] = %g, want a large negative (masked)", i, j, got)
			}
		}
	}

	// The method wraps it into a [1,1,L,S] array, which the stub holds as host data.
	b := &fb{s: mlxgo.DefaultStream()}
	model := &DeepseekV3Model{args: parseDeepseek(t, nil)}
	arr := model.causalMask(b, L, offset)
	if b.err != nil {
		t.Fatalf("causalMask err = %v", b.err)
	}
	if want := []int{1, 1, L, S}; !shapeEq(arr.Shape(), want) {
		t.Errorf("shape = %v, want %v", arr.Shape(), want)
	}
}

func shapeEq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestDeepseekYarnFreqs checks the host yarn build: the table has
// qk_rope_head_dim/2 finite positive entries, and the mscale follows the
// reference ratio (1 when mscale_all_dim equals the plain mscale of 1, and
// 0.1*ln(factor)+1 when mscale_all_dim is zero).
func TestDeepseekYarnFreqs(t *testing.T) {
	mk := func(mscaleAllDim float64) *DeepseekV3Args {
		return parseDeepseek(t, map[string]any{"rope_scaling": map[string]any{
			"type": "yarn", "factor": 40.0, "mscale_all_dim": mscaleAllDim,
			"original_max_position_embeddings": 4096, "beta_fast": 32.0, "beta_slow": 1.0,
		}})
	}

	a := mk(1.0)
	freqs, mscale := deepseekYarnFreqs(a)
	if len(freqs) != a.QKRopeHeadDim/2 {
		t.Fatalf("len(freqs) = %d, want %d", len(freqs), a.QKRopeHeadDim/2)
	}
	for i, f := range freqs {
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) || f <= 0 {
			t.Errorf("freqs[%d] = %g, want finite positive", i, f)
		}
	}
	if math.Abs(mscale-1.0) > 1e-9 {
		t.Errorf("mscale (all_dim=1) = %g, want 1.0", mscale)
	}

	_, mscale0 := deepseekYarnFreqs(mk(0.0))
	want := 0.1*math.Log(40.0) + 1.0
	if math.Abs(mscale0-want) > 1e-9 {
		t.Errorf("mscale (all_dim=0) = %g, want %g", mscale0, want)
	}
}

func BenchmarkDeepseekYarnFreqs(b *testing.B) {
	a, _ := ParseDeepseekV3Args(baseDeepseekConfig(map[string]any{"rope_scaling": map[string]any{
		"type": "yarn", "factor": 40.0, "mscale_all_dim": 1.0,
		"original_max_position_embeddings": 4096, "beta_fast": 32.0, "beta_slow": 1.0,
	}}))
	b.ReportAllocs()
	for b.Loop() {
		deepseekYarnFreqs(a)
	}
}

func BenchmarkDeepseekCausalMask(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		causalAdditiveMask(128, 0)
	}
}
