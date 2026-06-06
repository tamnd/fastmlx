// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"errors"
	"strings"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// parseQwen3Next builds args from the shared base config plus overrides.
func parseQwen3Next(t *testing.T, overrides map[string]any) *Qwen3NextArgs {
	t.Helper()
	a, err := ParseQwen3NextArgs(baseQwen3NextConfig(overrides))
	if err != nil {
		t.Fatalf("ParseQwen3NextArgs: %v", err)
	}
	return a
}

// hostArray builds a real (host-side) zero array of the given shape, the stub
// allocates without a kernel so helpers can run up to their first device op.
func hostArray(t *testing.T, shape ...int) *mlxgo.Array {
	t.Helper()
	a, err := mlxgo.Zeros(mlxgo.Float32, shape...)
	if err != nil {
		t.Fatalf("Zeros: %v", err)
	}
	return a
}

func TestNewQwen3NextModelWiresWeights(t *testing.T) {
	a := parseQwen3Next(t, nil)
	m, err := NewQwen3NextModel(a, fabricateWeights(t, a.WeightNames()))
	if err != nil {
		t.Fatalf("NewQwen3NextModel: %v", err)
	}
	if len(m.layers) != a.NumLayers() {
		t.Fatalf("layers = %d, want %d", len(m.layers), a.NumLayers())
	}
	if m.lmHead == nil {
		t.Fatal("lmHead is nil, want an untied head for this config")
	}
	// Base config: full_attention_interval 4 over 4 layers, so layers 0..2 are
	// linear and layer 3 is full attention; decoder_sparse_step 1 makes every
	// layer a mixture.
	for i, layer := range m.layers {
		wantLinear := a.IsLinear(i)
		if layer.isLinear != wantLinear {
			t.Errorf("layer %d isLinear = %v, want %v", i, layer.isLinear, wantLinear)
		}
		if layer.isLinear {
			if layer.linear.inProjQKVZ == nil || layer.linear.conv1d == nil {
				t.Errorf("layer %d (linear) missing gated delta net weights", i)
			}
			if layer.attn.qProj != nil {
				t.Errorf("layer %d (linear) wired attention weights", i)
			}
		} else {
			if layer.attn.qProj == nil || layer.attn.qNorm == nil {
				t.Errorf("layer %d (attention) missing attention weights", i)
			}
			if layer.linear.inProjQKVZ != nil {
				t.Errorf("layer %d (attention) wired gated delta net weights", i)
			}
		}
		if !layer.mlp.isMoE {
			t.Errorf("layer %d mlp.isMoE = false, want true (decoder_sparse_step 1)", i)
		}
		if layer.mlp.switchGate == nil || layer.mlp.sharedExpertGate == nil {
			t.Errorf("layer %d missing mixture weights", i)
		}
	}
}

func TestNewQwen3NextModelTiedHead(t *testing.T) {
	a := parseQwen3Next(t, map[string]any{"tie_word_embeddings": true})
	m, err := NewQwen3NextModel(a, fabricateWeights(t, a.WeightNames()))
	if err != nil {
		t.Fatalf("NewQwen3NextModel: %v", err)
	}
	if m.lmHead != nil {
		t.Fatal("lmHead is non-nil, want a tied head reusing the embedding table")
	}
}

func TestNewQwen3NextModelMissingWeight(t *testing.T) {
	a := parseQwen3Next(t, nil)
	_, err := NewQwen3NextModel(a, map[string]*mlxgo.Array{})
	if err == nil {
		t.Fatal("NewQwen3NextModel accepted an empty weight map")
	}
	if !strings.Contains(err.Error(), "missing weight") {
		t.Fatalf("error %q is not the missing-weight report", err)
	}
}

func TestNewQwen3NextModelRopeScalingRejected(t *testing.T) {
	a := parseQwen3Next(t, map[string]any{
		"rope_scaling": map[string]any{"type": "linear", "factor": 2.0},
	})
	_, err := NewQwen3NextModel(a, fabricateWeights(t, a.WeightNames()))
	if err == nil {
		t.Fatal("NewQwen3NextModel accepted a config with rope_scaling")
	}
	if !strings.Contains(err.Error(), "rope_scaling") {
		t.Fatalf("error %q does not name the unsupported rope_scaling", err)
	}
}

func TestNewQwen3NextModelAttentionBias(t *testing.T) {
	a := parseQwen3Next(t, map[string]any{"attention_bias": true})
	m, err := NewQwen3NextModel(a, fabricateWeights(t, a.WeightNames()))
	if err != nil {
		t.Fatalf("NewQwen3NextModel: %v", err)
	}
	for i, layer := range m.layers {
		if layer.isLinear {
			continue
		}
		if layer.attn.qBias == nil || layer.attn.oBias == nil {
			t.Errorf("layer %d missing attention biases", i)
		}
	}
}

func newQwen3NextCaches(n int) []*KVTensorCache {
	caches := make([]*KVTensorCache, n)
	for i := range caches {
		caches[i] = &KVTensorCache{}
	}
	return caches
}

func TestQwen3NextForwardGraceful(t *testing.T) {
	a := parseQwen3Next(t, nil)
	m, err := NewQwen3NextModel(a, fabricateWeights(t, a.WeightNames()))
	if err != nil {
		t.Fatalf("NewQwen3NextModel: %v", err)
	}
	cases := []struct {
		name   string
		tokens []int32
	}{
		{"prefill", []int32{1, 2, 3}},
		{"decode", []int32{7}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			caches := newQwen3NextCaches(a.NumLayers())
			_, err := m.Forward(tc.tokens, caches, nil)
			if !errors.Is(err, mlxgo.ErrMLXUnavailable) {
				t.Fatalf("Forward err = %v, want ErrMLXUnavailable", err)
			}
		})
	}
}

func TestQwen3NextForwardCacheMismatch(t *testing.T) {
	a := parseQwen3Next(t, nil)
	m, err := NewQwen3NextModel(a, fabricateWeights(t, a.WeightNames()))
	if err != nil {
		t.Fatalf("NewQwen3NextModel: %v", err)
	}
	_, err = m.Forward([]int32{1}, newQwen3NextCaches(a.NumLayers()-1), nil)
	if err == nil || errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("Forward err = %v, want a cache-count error", err)
	}
	if !strings.Contains(err.Error(), "caches") {
		t.Fatalf("error %q does not report the cache mismatch", err)
	}
}

func TestComputeGGraceful(t *testing.T) {
	a := parseQwen3Next(t, nil)
	nv := a.LinearNumValueHeads
	b := &fb{}
	aLog := hostArray(t, nv)
	dtBias := hostArray(t, nv)
	alpha := hostArray(t, 1, 2, nv)
	if got := b.computeG(aLog, alpha, dtBias); got != nil {
		t.Fatalf("computeG result = %v, want nil on the stub", got)
	}
	if !errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("computeG err = %v, want ErrMLXUnavailable", b.err)
	}
}

func TestDepthwiseConv1dGraceful(t *testing.T) {
	a := parseQwen3Next(t, nil)
	convDim := a.ConvDim()
	kSize := a.LinearConvKernelDim
	L := 3
	b := &fb{}
	x := hostArray(t, 1, kSize-1+L, convDim)
	weight := hostArray(t, convDim, kSize, 1)
	if got := b.depthwiseConv1d(x, weight, convDim, kSize, L); got != nil {
		t.Fatalf("depthwiseConv1d result = %v, want nil on the stub", got)
	}
	if !errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("depthwiseConv1d err = %v, want ErrMLXUnavailable", b.err)
	}
}

func TestGatedDeltaNetGraceful(t *testing.T) {
	a := parseQwen3Next(t, nil)
	m, err := NewQwen3NextModel(a, fabricateWeights(t, a.WeightNames()))
	if err != nil {
		t.Fatalf("NewQwen3NextModel: %v", err)
	}
	// Layer 0 is a linear (gated delta net) layer in the base config.
	var linear *qwen3NextLinear
	for i := range m.layers {
		if m.layers[i].isLinear {
			linear = &m.layers[i].linear
			break
		}
	}
	if linear == nil {
		t.Fatal("no linear layer in the base config")
	}
	for _, L := range []int{1, 4} {
		b := &fb{}
		x := hostArray(t, 1, L, a.HiddenSize)
		cache := &KVTensorCache{}
		if got := b.gatedDeltaNet(x, linear, a, cache, nil, 1, L); got != nil {
			t.Fatalf("L=%d gatedDeltaNet result = %v, want nil on the stub", L, got)
		}
		if !errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
			t.Fatalf("L=%d gatedDeltaNet err = %v, want ErrMLXUnavailable", L, b.err)
		}
	}
}

func TestSSMLeftPadMask(t *testing.T) {
	// A two-row ragged prefill: row 0 unpadded, row 1 left-padded by 2, block length
	// 4. The mask is 1 at a real position (pos >= leftPad) and 0 at the front padding,
	// shaped [batch, L, 1] so it broadcasts over the convolution channels.
	leftPad := []int{0, 2}
	const L = 4
	m, err := ssmLeftPadMask(leftPad, L, mlxgo.DefaultStream())
	if err != nil {
		t.Fatalf("ssmLeftPadMask: %v", err)
	}
	want := []int{2, L, 1}
	got := m.Shape()
	if len(got) != len(want) {
		t.Fatalf("shape = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("shape = %v, want %v", got, want)
		}
	}
	data, err := m.ToFloat32()
	if err != nil {
		t.Fatalf("ToFloat32: %v", err)
	}
	for b, pad := range leftPad {
		for pos := range L {
			want := float32(0)
			if pos >= pad {
				want = 1
			}
			if data[b*L+pos] != want {
				t.Errorf("row %d pos %d = %g, want %g", b, pos, data[b*L+pos], want)
			}
		}
	}
}

func TestQwen3NextAttentionGraceful(t *testing.T) {
	a := parseQwen3Next(t, nil)
	m, err := NewQwen3NextModel(a, fabricateWeights(t, a.WeightNames()))
	if err != nil {
		t.Fatalf("NewQwen3NextModel: %v", err)
	}
	var attn *qwen3NextAttn
	for i := range m.layers {
		if !m.layers[i].isLinear {
			attn = &m.layers[i].attn
			break
		}
	}
	if attn == nil {
		t.Fatal("no attention layer in the base config")
	}
	b := &fb{}
	L := 2
	x := hostArray(t, 1, L, a.HiddenSize)
	cache := &KVTensorCache{}
	if got := b.qwen3NextAttention(x, attn, a, cache, "causal", nil, nil, 1, L); got != nil {
		t.Fatalf("qwen3NextAttention result = %v, want nil on the stub", got)
	}
	if !errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("qwen3NextAttention err = %v, want ErrMLXUnavailable", b.err)
	}
}

func TestQwen3NextMoEGraceful(t *testing.T) {
	a := parseQwen3Next(t, nil)
	m, err := NewQwen3NextModel(a, fabricateWeights(t, a.WeightNames()))
	if err != nil {
		t.Fatalf("NewQwen3NextModel: %v", err)
	}
	mlp := &m.layers[0].mlp
	if !mlp.isMoE {
		t.Fatal("layer 0 is not a mixture in the base config")
	}
	b := &fb{}
	L := 2
	x := hostArray(t, 1, L, a.HiddenSize)
	if got := b.qwen3NextMoE(x, mlp, a, 1, L); got != nil {
		t.Fatalf("qwen3NextMoE result = %v, want nil on the stub", got)
	}
	if !errors.Is(b.err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("qwen3NextMoE err = %v, want ErrMLXUnavailable", b.err)
	}
}

func TestQwen3NextCacheState(t *testing.T) {
	conv := hostArray(t, 1, 3, 8)
	state := hostArray(t, 1, 8, 16, 16)
	c := &KVTensorCache{Offset: 5}
	if c.ConvState() != nil || c.SSMState() != nil {
		t.Fatal("fresh cache should have nil gated delta net state")
	}
	c.SetState(conv, state, 4)
	if c.ConvState() != conv || c.SSMState() != state {
		t.Fatal("SetState did not record the conv and recurrent state")
	}
	if c.Offset != 9 {
		t.Fatalf("Offset = %d, want 9 (5 + 4)", c.Offset)
	}
}

func BenchmarkNewQwen3NextModel(b *testing.B) {
	a, err := ParseQwen3NextArgs(baseQwen3NextConfig(nil))
	if err != nil {
		b.Fatalf("ParseQwen3NextArgs: %v", err)
	}
	names := a.WeightNames()
	weights := make(map[string]*mlxgo.Array, len(names))
	for _, n := range names {
		weights[n], _ = mlxgo.Zeros(mlxgo.Float32, 1)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := NewQwen3NextModel(a, weights); err != nil {
			b.Fatalf("NewQwen3NextModel: %v", err)
		}
	}
}
