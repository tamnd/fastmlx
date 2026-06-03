// SPDX-License-Identifier: MIT OR Apache-2.0

package scheduler

import (
	"encoding/json"
	"os"
	"testing"
)

type transientFixture struct {
	Steps []struct {
		In              [2]int  `json:"in"`
		BytesPerToken   float64 `json:"bytes_per_token"`
		Samples         int     `json:"samples"`
		LastDeltaBytes  int     `json:"last_delta_bytes"`
		LastNTokens     int     `json:"last_n_tokens"`
		Predict128      int     `json:"predict_128"`
		Predict128SF1_0 int     `json:"predict_128_sf_1_0"`
		Predict0        int     `json:"predict_0"`
	} `json:"steps"`
	PredictFresh []int `json:"predict_fresh"`
	AfterReset   struct {
		BytesPerToken  float64 `json:"bytes_per_token"`
		Samples        int     `json:"samples"`
		LastDeltaBytes int     `json:"last_delta_bytes"`
		LastNTokens    int     `json:"last_n_tokens"`
		Predict128     int     `json:"predict_128"`
	} `json:"after_reset"`
}

func loadTransientFixture(t *testing.T) transientFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/prefill_transient.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f transientFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return f
}

func TestPrefillTransientSequence(t *testing.T) {
	f := loadTransientFixture(t)
	tr := NewPrefillTransientTracker("qwen3")
	for i, step := range f.Steps {
		tr.Update(step.In[0], step.In[1])
		if got := tr.BytesPerToken(); got != step.BytesPerToken {
			t.Errorf("step %d BytesPerToken = %v, want %v", i, got, step.BytesPerToken)
		}
		if got := tr.Samples(); got != step.Samples {
			t.Errorf("step %d Samples = %d, want %d", i, got, step.Samples)
		}
		if got := tr.LastDeltaBytes(); got != step.LastDeltaBytes {
			t.Errorf("step %d LastDeltaBytes = %d, want %d", i, got, step.LastDeltaBytes)
		}
		if got := tr.LastNTokens(); got != step.LastNTokens {
			t.Errorf("step %d LastNTokens = %d, want %d", i, got, step.LastNTokens)
		}
		if got := tr.PredictDefault(128); got != step.Predict128 {
			t.Errorf("step %d PredictDefault(128) = %d, want %d", i, got, step.Predict128)
		}
		if got := tr.Predict(128, 1.0); got != step.Predict128SF1_0 {
			t.Errorf("step %d Predict(128, 1.0) = %d, want %d", i, got, step.Predict128SF1_0)
		}
		if got := tr.PredictDefault(0); got != step.Predict0 {
			t.Errorf("step %d PredictDefault(0) = %d, want %d", i, got, step.Predict0)
		}
	}
}

func TestPrefillTransientFreshPredict(t *testing.T) {
	f := loadTransientFixture(t)
	fresh := NewPrefillTransientTracker("")
	got := []int{fresh.PredictDefault(128), fresh.Predict(0, 2.0)}
	if got[0] != f.PredictFresh[0] || got[1] != f.PredictFresh[1] {
		t.Errorf("fresh predict = %v, want %v", got, f.PredictFresh)
	}
}

func TestPrefillTransientReset(t *testing.T) {
	f := loadTransientFixture(t)
	tr := NewPrefillTransientTracker("qwen3")
	tr.Update(100, 200000)
	tr.Update(200, 500000)
	tr.Reset()
	if tr.BytesPerToken() != f.AfterReset.BytesPerToken ||
		tr.Samples() != f.AfterReset.Samples ||
		tr.LastDeltaBytes() != f.AfterReset.LastDeltaBytes ||
		tr.LastNTokens() != f.AfterReset.LastNTokens ||
		tr.PredictDefault(128) != f.AfterReset.Predict128 {
		t.Errorf("after reset: bpt=%v samples=%d delta=%d ntok=%d predict=%d, want %+v",
			tr.BytesPerToken(), tr.Samples(), tr.LastDeltaBytes(), tr.LastNTokens(), tr.PredictDefault(128), f.AfterReset)
	}
}

func BenchmarkPrefillTransientUpdate(b *testing.B) {
	tr := NewPrefillTransientTracker("qwen3")
	b.ReportAllocs()
	n := 0
	for b.Loop() {
		n++
		tr.Update(100+n%64, 200000+n)
		_ = tr.PredictDefault(128)
	}
}
