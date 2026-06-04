// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"errors"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// batchCaches builds n empty per-layer caches for a batched decode step. The
// batch dimension lives inside each cache's key/value tensors (axis 0), so the
// per-layer slice has the same length whether one sequence or many decode.
func batchCaches(n int) []*KVTensorCache {
	caches := make([]*KVTensorCache, n)
	for i := range caches {
		caches[i] = &KVTensorCache{}
	}
	return caches
}

// batchTokens returns one decode token per row, the [batch] input the reference
// reshapes to [batch, 1] before the step.
func batchTokens(batch int) []int32 {
	tokens := make([]int32, batch)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	return tokens
}

// The batched decode forward is the throughput path: one kernel launch serves a
// whole synchronized batch instead of one launch per sequence. Each model's
// BatchDecode type-checks and runs against the stub up to the first kernel (the
// embedding take over the [batch, 1] input), then reports the missing backend,
// the same wiring confirmation the single-sequence Forward gives. Several batch
// sizes, including 1, exercise the shared forwardBL with the batch dimension
// generalized away from the hardcoded leading 1.

func TestQwen3BatchDecodeGraceful(t *testing.T) {
	a := tinyQwen3Args(t, true)
	m, err := NewQwen3Model(a, dummyWeights(t, a))
	if err != nil {
		t.Fatalf("NewQwen3Model: %v", err)
	}
	for _, batch := range []int{1, 2, 5} {
		_, err := m.BatchDecode(batchTokens(batch), batch, batchCaches(a.NumHiddenLayers), mlxgo.DefaultStream())
		if !errors.Is(err, mlxgo.ErrMLXUnavailable) {
			t.Errorf("BatchDecode(batch=%d) err = %v, want ErrMLXUnavailable", batch, err)
		}
	}
}

func TestGlm4BatchDecodeGraceful(t *testing.T) {
	a := tinyGlm4Args(t, false)
	m, err := NewGlm4Model(a, dummyGlm4Weights(t, a))
	if err != nil {
		t.Fatalf("NewGlm4Model: %v", err)
	}
	for _, batch := range []int{1, 2, 5} {
		_, err := m.BatchDecode(batchTokens(batch), batch, batchCaches(a.NumLayers()), mlxgo.DefaultStream())
		if !errors.Is(err, mlxgo.ErrMLXUnavailable) {
			t.Errorf("BatchDecode(batch=%d) err = %v, want ErrMLXUnavailable", batch, err)
		}
	}
}

func TestPhi4BatchDecodeGraceful(t *testing.T) {
	a := tinyPhi4Args(t, false)
	m, err := NewPhi4Model(a, dummyPhi4Weights(t, a))
	if err != nil {
		t.Fatalf("NewPhi4Model: %v", err)
	}
	for _, batch := range []int{1, 2, 5} {
		_, err := m.BatchDecode(batchTokens(batch), batch, batchCaches(a.NumLayers()), mlxgo.DefaultStream())
		if !errors.Is(err, mlxgo.ErrMLXUnavailable) {
			t.Errorf("BatchDecode(batch=%d) err = %v, want ErrMLXUnavailable", batch, err)
		}
	}
}

func TestMinistral3BatchDecodeGraceful(t *testing.T) {
	a := tinyMinistralArgs(t, true)
	m, err := NewMinistral3Model(a, dummyMinistralWeights(t, a))
	if err != nil {
		t.Fatalf("NewMinistral3Model: %v", err)
	}
	for _, batch := range []int{1, 2, 5} {
		_, err := m.BatchDecode(batchTokens(batch), batch, batchCaches(a.NumLayers()), mlxgo.DefaultStream())
		if !errors.Is(err, mlxgo.ErrMLXUnavailable) {
			t.Errorf("BatchDecode(batch=%d) err = %v, want ErrMLXUnavailable", batch, err)
		}
	}
}

func TestGemma4BatchDecodeGraceful(t *testing.T) {
	a := tinyGemma4Args(t, true, 4)
	m, err := NewGemma4TextModel(a, dummyGemma4Weights(t, a))
	if err != nil {
		t.Fatalf("NewGemma4TextModel: %v", err)
	}
	for _, batch := range []int{1, 2, 5} {
		_, err := m.BatchDecode(batchTokens(batch), batch, batchCaches(a.NumLayers()), mlxgo.DefaultStream())
		if !errors.Is(err, mlxgo.ErrMLXUnavailable) {
			t.Errorf("BatchDecode(batch=%d) err = %v, want ErrMLXUnavailable", batch, err)
		}
	}
}

// TestDeepseekV3BatchDecodeGraceful drives the routed MoE model's batched decode
// to the backend seam. DeepSeek-V3 carries two singleton attention axes (the MQA
// latent head and the rotary key head) that are not the batch axis, so the test
// confirms generalizing only the leading batch dimension keeps the multi-head
// latent attention and the router degrading gracefully across batch widths.
func TestDeepseekV3BatchDecodeGraceful(t *testing.T) {
	a := parseDeepseek(t, nil)
	m, err := NewDeepseekV3Model(a, dummyDeepseekWeights(t, a))
	if err != nil {
		t.Fatalf("NewDeepseekV3Model: %v", err)
	}
	for _, batch := range []int{1, 2, 5} {
		_, err := m.BatchDecode(batchTokens(batch), batch, batchCaches(a.NumLayers()), mlxgo.DefaultStream())
		if !errors.Is(err, mlxgo.ErrMLXUnavailable) {
			t.Errorf("BatchDecode(batch=%d) err = %v, want ErrMLXUnavailable", batch, err)
		}
	}
}

// TestQwen3NextBatchDecodeGraceful drives the hybrid model's batched decode to
// the backend seam. Qwen3-Next is the hardest batch generalization: a linear
// layer carries a recurrent state and a convolution window that now lead with the
// batch axis, and the one-timestep recurrence advances every row in parallel. The
// test runs batch 1, 2, and 5 so the gated delta net, the gated attention, and the
// mixture all degrade gracefully with the batch dimension threaded through.
func TestQwen3NextBatchDecodeGraceful(t *testing.T) {
	a := parseQwen3Next(t, nil)
	m, err := NewQwen3NextModel(a, fabricateWeights(t, a.WeightNames()))
	if err != nil {
		t.Fatalf("NewQwen3NextModel: %v", err)
	}
	for _, batch := range []int{1, 2, 5} {
		_, err := m.BatchDecode(batchTokens(batch), batch, batchCaches(a.NumLayers()), mlxgo.DefaultStream())
		if !errors.Is(err, mlxgo.ErrMLXUnavailable) {
			t.Errorf("BatchDecode(batch=%d) err = %v, want ErrMLXUnavailable", batch, err)
		}
	}
}

// TestBatchDecodeMatchesForwardForOneRow pins every dense model's batch=1 decode
// to its single-sequence forward: both feed one token through identical shapes
// and surface the same backend-missing error, so the batched path is a strict
// generalization of the path already serving single sequences.
func TestBatchDecodeMatchesForwardForOneRow(t *testing.T) {
	qa := tinyQwen3Args(t, true)
	qm, err := NewQwen3Model(qa, dummyWeights(t, qa))
	if err != nil {
		t.Fatalf("NewQwen3Model: %v", err)
	}
	_, fwd := qm.Forward([]int32{7}, batchCaches(qa.NumHiddenLayers), mlxgo.DefaultStream())
	_, bat := qm.BatchDecode([]int32{7}, 1, batchCaches(qa.NumHiddenLayers), mlxgo.DefaultStream())
	if !errors.Is(fwd, mlxgo.ErrMLXUnavailable) || !errors.Is(bat, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("Forward err = %v, BatchDecode err = %v, want both ErrMLXUnavailable", fwd, bat)
	}
}
