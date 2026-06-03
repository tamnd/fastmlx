// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/tamnd/fastmlx/eval"
)

type batchCase struct {
	V  int  `json:"v"`
	Ok bool `json:"ok"`
}

type benchmarkSetCase struct {
	Set map[string]int `json:"set"`
	Ok  bool           `json:"ok"`
}

type questionIn struct {
	QuestionID   string  `json:"question_id"`
	Correct      bool    `json:"correct"`
	Expected     string  `json:"expected"`
	Predicted    string  `json:"predicted"`
	QuestionText string  `json:"question_text"`
	RawResponse  string  `json:"raw_response"`
	Category     string  `json:"category"`
	TimeSeconds  float64 `json:"time_seconds"`
}

type resultIn struct {
	ModelID         string             `json:"model_id"`
	BenchmarkName   string             `json:"benchmark_name"`
	Accuracy        float64            `json:"accuracy"`
	ThinkingUsed    bool               `json:"thinking_used"`
	TotalQuestions  int                `json:"total_questions"`
	CorrectCount    int                `json:"correct_count"`
	TimeSeconds     float64            `json:"time_seconds"`
	QuestionResults []questionIn       `json:"question_results"`
	CategoryScores  map[string]float64 `json:"category_scores"`
}

type accuracyBenchFixture struct {
	ValidBenchmarks []string           `json:"valid_benchmarks"`
	Batch           []batchCase        `json:"batch"`
	Benchmarks      []benchmarkSetCase `json:"benchmarks"`
	ResultsIn       []resultIn         `json:"results_in"`
	ResultData      []map[string]any   `json:"result_data"`
}

func loadAccuracyBench(t *testing.T) accuracyBenchFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/accuracybench.json")
	if err != nil {
		t.Fatal(err)
	}
	var f accuracyBenchFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func jsonRoundTrip(t *testing.T, v any) map[string]any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func (r resultIn) toResult() eval.BenchmarkResult {
	qrs := make([]eval.QuestionResult, len(r.QuestionResults))
	for i, q := range r.QuestionResults {
		qrs[i] = eval.QuestionResult{
			QuestionID:   q.QuestionID,
			Correct:      q.Correct,
			Expected:     q.Expected,
			Predicted:    q.Predicted,
			TimeSeconds:  q.TimeSeconds,
			QuestionText: q.QuestionText,
			RawResponse:  q.RawResponse,
			Category:     q.Category,
		}
	}
	return eval.BenchmarkResult{
		BenchmarkName:   r.BenchmarkName,
		Accuracy:        r.Accuracy,
		TotalQuestions:  r.TotalQuestions,
		CorrectCount:    r.CorrectCount,
		TimeSeconds:     r.TimeSeconds,
		QuestionResults: qrs,
		CategoryScores:  r.CategoryScores,
		ThinkingUsed:    r.ThinkingUsed,
	}
}

func TestValidBenchmarksParity(t *testing.T) {
	if !reflect.DeepEqual(ValidBenchmarks, loadAccuracyBench(t).ValidBenchmarks) {
		t.Errorf("ValidBenchmarks = %v, want %v", ValidBenchmarks, loadAccuracyBench(t).ValidBenchmarks)
	}
}

func TestValidBenchmarksMatchRegistry(t *testing.T) {
	reg := eval.BenchmarkNames() // sorted
	got := append([]string(nil), ValidBenchmarks...)
	// Same membership as the eval registry, order aside.
	set := map[string]bool{}
	for _, n := range got {
		set[n] = true
	}
	for _, n := range reg {
		if !set[n] {
			t.Errorf("registry benchmark %q missing from ValidBenchmarks", n)
		}
	}
	if len(got) != len(reg) {
		t.Errorf("ValidBenchmarks has %d entries, registry has %d", len(got), len(reg))
	}
}

func TestValidBatchSizeParity(t *testing.T) {
	for i, c := range loadAccuracyBench(t).Batch {
		if got := ValidBatchSize(c.V); got != c.Ok {
			t.Errorf("batch case %d (v=%d) = %v, want %v", i, c.V, got, c.Ok)
		}
	}
}

func TestValidBenchmarkSetParity(t *testing.T) {
	for i, c := range loadAccuracyBench(t).Benchmarks {
		if got := ValidBenchmarkSet(c.Set); got != c.Ok {
			t.Errorf("benchmark-set case %d (%v) = %v, want %v", i, c.Set, got, c.Ok)
		}
	}
}

func TestResultDataParity(t *testing.T) {
	f := loadAccuracyBench(t)
	for i, in := range f.ResultsIn {
		got := ResultData(in.toResult(), in.ModelID)
		want := f.ResultData[i]
		if !reflect.DeepEqual(jsonRoundTrip(t, got), want) {
			gb, _ := json.Marshal(got)
			wb, _ := json.Marshal(want)
			t.Errorf("ResultData case %d =\n%s\nwant\n%s", i, gb, wb)
		}
	}
}

func BenchmarkResultData(b *testing.B) {
	result := eval.BenchmarkResult{
		BenchmarkName:  "mmlu",
		Accuracy:       0.736842105,
		TotalQuestions: 57,
		CorrectCount:   42,
		TimeSeconds:    123.4567,
		QuestionResults: []eval.QuestionResult{
			{QuestionID: "0", Correct: true, Expected: "A", Predicted: "A", QuestionText: "Q?", RawResponse: "A", Category: "math", TimeSeconds: 1.23456},
		},
		CategoryScores: map[string]float64{"math": 1.0, "law": 0.66665},
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = ResultData(result, "qwen3-4b")
	}
}
