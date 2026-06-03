// SPDX-License-Identifier: MIT OR Apache-2.0

package metrics

import (
	"encoding/json"
	"os"
	"testing"
)

// fixture mirrors metrics/testdata/servermetrics.json, captured from the
// reference ServerMetrics so the Go port reproduces it byte for byte.
type fixture struct {
	BuildSnapshot []struct {
		In  []float64    `json:"in"`
		Out snapshotJSON `json:"out"`
	} `json:"build_snapshot"`
	Sequence struct {
		Calls []struct {
			PromptTokens       int     `json:"prompt_tokens"`
			CompletionTokens   int     `json:"completion_tokens"`
			CachedTokens       int     `json:"cached_tokens"`
			PrefillDuration    float64 `json:"prefill_duration"`
			GenerationDuration float64 `json:"generation_duration"`
			ModelID            string  `json:"model_id"`
		} `json:"calls"`
		Snapshots []struct {
			ModelID string       `json:"model_id"`
			Scope   string       `json:"scope"`
			Out     snapshotJSON `json:"out"`
		} `json:"snapshots"`
	} `json:"sequence"`
}

// snapshotJSON decodes the reference snapshot dict. The field order in the
// struct matches the reference dict order so the json tags line up one to one
// with Snapshot.
type snapshotJSON struct {
	TotalTokensServed     int     `json:"total_tokens_served"`
	TotalCachedTokens     int     `json:"total_cached_tokens"`
	CacheEfficiency       float64 `json:"cache_efficiency"`
	TotalPromptTokens     int     `json:"total_prompt_tokens"`
	TotalCompletionTokens int     `json:"total_completion_tokens"`
	TotalRequests         int     `json:"total_requests"`
	AvgPrefillTPS         float64 `json:"avg_prefill_tps"`
	AvgGenerationTPS      float64 `json:"avg_generation_tps"`
	UptimeSeconds         float64 `json:"uptime_seconds"`
}

func (s snapshotJSON) snapshot() Snapshot {
	return Snapshot(s)
}

func loadFixture(t *testing.T) fixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/servermetrics.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f fixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return f
}

func TestBuildSnapshot(t *testing.T) {
	f := loadFixture(t)
	for i, tc := range f.BuildSnapshot {
		in := tc.In
		c := counters{
			promptTokens:       int(in[0]),
			completionTokens:   int(in[1]),
			cachedTokens:       int(in[2]),
			requests:           int(in[3]),
			prefillDuration:    in[4],
			generationDuration: in[5],
		}
		got := buildSnapshot(c, in[6])
		if want := tc.Out.snapshot(); got != want {
			t.Errorf("build_snapshot[%d] in=%v\n got=%+v\nwant=%+v", i, in, got, want)
		}
	}
}

func TestRecordAndGetSnapshot(t *testing.T) {
	f := loadFixture(t)
	m := NewServerMetrics()
	for _, c := range f.Sequence.Calls {
		m.RecordRequestComplete(c.PromptTokens, c.CompletionTokens, c.CachedTokens, c.PrefillDuration, c.GenerationDuration, c.ModelID)
	}
	// The reference fixture used uptime 1000.0 for every query.
	const uptime = 1000.0
	for i, q := range f.Sequence.Snapshots {
		got := m.GetSnapshot(q.ModelID, q.Scope, uptime)
		if want := q.Out.snapshot(); got != want {
			t.Errorf("snapshot[%d] model=%q scope=%q\n got=%+v\nwant=%+v", i, q.ModelID, q.Scope, got, want)
		}
	}
}

// TestRound1Edge pins the Python-faithful rounding on the 0.05 case, where the
// naive x*10 trick would yield 0.0 instead of 0.1.
func TestRound1Edge(t *testing.T) {
	if got := round1(0.05); got != 0.1 {
		t.Errorf("round1(0.05) = %v, want 0.1", got)
	}
	if got := round1(193.75); got != 193.8 {
		t.Errorf("round1(193.75) = %v, want 193.8", got)
	}
}

func BenchmarkGetSnapshot(b *testing.B) {
	m := NewServerMetrics()
	for range 200 {
		m.RecordRequestComplete(100, 40, 10, 0.5, 1.0, "qwen3")
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = m.GetSnapshot("qwen3", "session", 1234.5)
	}
}
