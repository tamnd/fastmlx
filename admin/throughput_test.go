// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type metricsIn struct {
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	StartTime        float64 `json:"start_time"`
	FirstTokenTime   float64 `json:"first_token_time"`
	EndTime          float64 `json:"end_time"`
	PeakMemory       int     `json:"peak_memory"`
	CachedTokens     int     `json:"cached_tokens"`
}

type metricsCase struct {
	In  metricsIn      `json:"in"`
	Out map[string]any `json:"out"`
}

type quantCase struct {
	Name string `json:"name"`
	Out  string `json:"out"`
}

type cleanCase struct {
	ModelID string `json:"model_id"`
	Out     string `json:"out"`
}

type intListValidationCase struct {
	In  []int `json:"in"`
	Ok  bool  `json:"ok"`
	Out []int `json:"out"`
}

type throughputFixture struct {
	ValidPromptLengths []int                   `json:"valid_prompt_lengths"`
	ValidBatchSizes    []int                   `json:"valid_batch_sizes"`
	Metrics            []metricsCase           `json:"metrics"`
	Quant              []quantCase             `json:"quant"`
	Clean              []cleanCase             `json:"clean"`
	PromptValidation   []intListValidationCase `json:"prompt_validation"`
	BatchValidation    []intListValidationCase `json:"batch_validation"`
}

func loadThroughput(t *testing.T) throughputFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/throughput.json")
	if err != nil {
		t.Fatal(err)
	}
	var f throughputFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestThroughputConstantsParity(t *testing.T) {
	f := loadThroughput(t)
	if !reflect.DeepEqual(ValidPromptLengths, f.ValidPromptLengths) {
		t.Errorf("ValidPromptLengths = %v, want %v", ValidPromptLengths, f.ValidPromptLengths)
	}
	if !reflect.DeepEqual(ValidThroughputBatchSizes, f.ValidBatchSizes) {
		t.Errorf("ValidThroughputBatchSizes = %v, want %v", ValidThroughputBatchSizes, f.ValidBatchSizes)
	}
}

func TestComputeSingleMetricsParity(t *testing.T) {
	for i, c := range loadThroughput(t).Metrics {
		got := ComputeSingleMetrics(c.In.PromptTokens, c.In.CompletionTokens,
			c.In.StartTime, c.In.FirstTokenTime, c.In.EndTime, c.In.PeakMemory, c.In.CachedTokens)
		if !reflect.DeepEqual(jsonRoundTrip(t, got), c.Out) {
			gb, _ := json.Marshal(got)
			wb, _ := json.Marshal(c.Out)
			t.Errorf("ComputeSingleMetrics case %d =\n%s\nwant\n%s", i, gb, wb)
		}
	}
}

func TestDetectQuantizationFromNameParity(t *testing.T) {
	for i, c := range loadThroughput(t).Quant {
		if got := DetectQuantizationFromName(c.Name); got != c.Out {
			t.Errorf("DetectQuantizationFromName case %d (%q) = %q, want %q", i, c.Name, got, c.Out)
		}
	}
}

func TestDetectQuantizationBits(t *testing.T) {
	if got := DetectQuantization(4, true, "ignored-name"); got != "4bit" {
		t.Errorf("DetectQuantization with bits = %q, want %q", got, "4bit")
	}
	if got := DetectQuantization(0, false, "Llama-3-fp16"); got != "fp16" {
		t.Errorf("DetectQuantization name fallback = %q, want %q", got, "fp16")
	}
}

func TestCleanModelNameParity(t *testing.T) {
	for i, c := range loadThroughput(t).Clean {
		if got := CleanModelName(c.ModelID); got != c.Out {
			t.Errorf("CleanModelName case %d (%q) = %q, want %q", i, c.ModelID, got, c.Out)
		}
	}
}

func TestValidatePromptLengthsParity(t *testing.T) {
	for i, c := range loadThroughput(t).PromptValidation {
		out, ok := ValidatePromptLengths(c.In)
		if ok != c.Ok {
			t.Errorf("ValidatePromptLengths case %d (%v): ok = %v, want %v", i, c.In, ok, c.Ok)
			continue
		}
		if ok && !reflect.DeepEqual(out, c.Out) {
			t.Errorf("ValidatePromptLengths case %d (%v) = %v, want %v", i, c.In, out, c.Out)
		}
	}
}

func TestValidateBatchSizesParity(t *testing.T) {
	for i, c := range loadThroughput(t).BatchValidation {
		out, ok := ValidateBatchSizes(c.In)
		if ok != c.Ok {
			t.Errorf("ValidateBatchSizes case %d (%v): ok = %v, want %v", i, c.In, ok, c.Ok)
			continue
		}
		if ok && !reflect.DeepEqual(out, c.Out) {
			t.Errorf("ValidateBatchSizes case %d (%v) = %v, want %v", i, c.In, out, c.Out)
		}
	}
}

func BenchmarkComputeSingleMetrics(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = ComputeSingleMetrics(1024, 128, 0.0, 0.25, 2.75, 8589934592, 0)
	}
}

func BenchmarkCleanModelName(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = CleanModelName("Qwen3-8B-MLX-bf16")
	}
}
