// SPDX-License-Identifier: MIT OR Apache-2.0

package bench

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type benchFixture struct {
	Single []struct {
		In struct {
			PromptTokens     int     `json:"prompt_tokens"`
			CompletionTokens int     `json:"completion_tokens"`
			StartTime        float64 `json:"start_time"`
			FirstTokenTime   float64 `json:"first_token_time"`
			EndTime          float64 `json:"end_time"`
			PeakMemory       int64   `json:"peak_memory"`
			CachedTokens     int     `json:"cached_tokens"`
		} `json:"in"`
		Out map[string]float64 `json:"out"`
	} `json:"single"`
	Batch   map[string]float64 `json:"batch"`
	BatchIn struct {
		Results []struct {
			CompletionTokens int     `json:"completion_tokens"`
			TTFTS            float64 `json:"ttft_s"`
			FirstTokenAbs    float64 `json:"first_token_abs"`
		} `json:"results"`
		PromptTokens int     `json:"prompt_tokens"`
		BatchSize    int     `json:"batch_size"`
		WallStart    float64 `json:"wall_start"`
		WallEnd      float64 `json:"wall_end"`
	} `json:"batch_in"`
	CleanName []struct {
		In  string `json:"in"`
		Out string `json:"out"`
	} `json:"clean_name"`
	DetectQuant []struct {
		In  string `json:"in"`
		Out string `json:"out"`
	} `json:"detect_quant"`
}

func loadFixture(t *testing.T) benchFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/bench.json")
	if err != nil {
		t.Fatal(err)
	}
	var f benchFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func approxEq(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func TestComputeSingleMetricsParity(t *testing.T) {
	fx := loadFixture(t)
	for i, c := range fx.Single {
		got := ComputeSingleMetrics(SingleInput{
			PromptTokens:     c.In.PromptTokens,
			CompletionTokens: c.In.CompletionTokens,
			StartTime:        c.In.StartTime,
			FirstTokenTime:   c.In.FirstTokenTime,
			EndTime:          c.In.EndTime,
			PeakMemory:       c.In.PeakMemory,
			CachedTokens:     c.In.CachedTokens,
		})
		checks := map[string]float64{
			"ttft_ms":           got.TTFTMs,
			"tpot_ms":           got.TPOTMs,
			"gen_tps":           got.GenTPS,
			"processing_tps":    got.ProcessingTPS,
			"e2e_latency_s":     got.E2ELatencyS,
			"total_throughput":  got.TotalThroughput,
			"peak_memory_bytes": float64(got.PeakMemoryBytes),
			"prompt_tokens":     float64(got.PromptTokens),
			"completion_tokens": float64(got.CompletionTokens),
			"cached_tokens":     float64(got.CachedTokens),
		}
		for k, v := range checks {
			if !approxEq(v, c.Out[k]) {
				t.Errorf("single[%d].%s = %v, want %v", i, k, v, c.Out[k])
			}
		}
	}
}

func TestBatchAggregateParity(t *testing.T) {
	fx := loadFixture(t)
	results := make([]BatchResult, len(fx.BatchIn.Results))
	for i, r := range fx.BatchIn.Results {
		results[i] = BatchResult{CompletionTokens: r.CompletionTokens, TTFTS: r.TTFTS, FirstTokenAbs: r.FirstTokenAbs}
	}
	got := BatchAggregate(results, fx.BatchIn.PromptTokens, fx.BatchIn.BatchSize, fx.BatchIn.WallStart, fx.BatchIn.WallEnd)
	checks := map[string]float64{
		"pp_tps":           got.PPTPS,
		"tg_tps":           got.TGTPS,
		"avg_ttft_ms":      got.AvgTTFTMs,
		"e2e_latency_s":    got.E2ELatencyS,
		"total_gen_tokens": float64(got.TotalGenTokens),
		"batch_size":       float64(got.BatchSize),
	}
	for k, v := range checks {
		if !approxEq(v, fx.Batch[k]) {
			t.Errorf("batch.%s = %v, want %v", k, v, fx.Batch[k])
		}
	}
}

func TestCleanModelNameParity(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.CleanName {
		if got := CleanModelName(c.In); got != c.Out {
			t.Errorf("CleanModelName(%q) = %q, want %q", c.In, got, c.Out)
		}
	}
}

func TestDetectQuantizationFromNameParity(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.DetectQuant {
		if got := DetectQuantizationFromName(c.In); got != c.Out {
			t.Errorf("DetectQuantizationFromName(%q) = %q, want %q", c.In, got, c.Out)
		}
	}
}

func BenchmarkComputeSingleMetrics(b *testing.B) {
	in := SingleInput{PromptTokens: 512, CompletionTokens: 128, StartTime: 100, FirstTokenTime: 100.25, EndTime: 102.75, PeakMemory: 8e9}
	b.ReportAllocs()
	for b.Loop() {
		_ = ComputeSingleMetrics(in)
	}
}
