// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

// This file holds the GPU-free result types and the pure scoring and
// aggregation a benchmark run folds over. Generating the model response and
// timing each batch are the caller's seams; given a response, turning it into a
// scored question and rolling a slice of those into a run summary is pure.

// QuestionResult is the outcome for a single benchmark question.
type QuestionResult struct {
	QuestionID   string  `json:"question_id"`
	Correct      bool    `json:"correct"`
	Expected     string  `json:"expected"`
	Predicted    string  `json:"predicted"`
	TimeSeconds  float64 `json:"time_seconds"`
	QuestionText string  `json:"question_text"`
	RawResponse  string  `json:"raw_response"`
	// Category is the per-item subject, or "" when the benchmark has none.
	Category string `json:"category"`
}

// BenchmarkResult is the aggregated outcome of a complete benchmark run.
type BenchmarkResult struct {
	BenchmarkName   string           `json:"benchmark_name"`
	Accuracy        float64          `json:"accuracy"`
	TotalQuestions  int              `json:"total_questions"`
	CorrectCount    int              `json:"correct_count"`
	TimeSeconds     float64          `json:"time_seconds"`
	QuestionResults []QuestionResult `json:"question_results"`
	// CategoryScores is nil when no question carried a category.
	CategoryScores map[string]float64 `json:"category_scores"`
	ThinkingUsed   bool               `json:"thinking_used"`
}

// ScoreQuestion turns one generated response into a scored question. It runs the
// benchmark's pure extract, check, and category steps and assembles the record
// the run loop appends, using the item's "id" when present and otherwise the
// batch index, and the item's "answer" as the expected value.
func ScoreQuestion(b Benchmark, item Item, idx int, response, promptText string, timeSeconds float64) QuestionResult {
	predicted := b.ExtractAnswer(response, item)
	correct := b.CheckAnswer(predicted, item)
	category := b.Category(item)

	qid := asStr(idx)
	if v, ok := item["id"]; ok {
		qid = asStr(v)
	}
	expected := ""
	if v, ok := item["answer"]; ok {
		expected = asStr(v)
	}

	return QuestionResult{
		QuestionID:   qid,
		Correct:      correct,
		Expected:     expected,
		Predicted:    predicted,
		TimeSeconds:  timeSeconds,
		QuestionText: promptText,
		RawResponse:  response,
		Category:     category,
	}
}

// Aggregate folds scored questions into a run summary: overall accuracy, the
// correct count, and a per-category breakdown when any question carried a
// category. A run with no questions reports zero accuracy.
func Aggregate(name string, results []QuestionResult, timeSeconds float64, thinkingUsed bool) BenchmarkResult {
	total := len(results)
	correct := 0
	categoryTotal := map[string]int{}
	categoryCorrect := map[string]int{}

	for _, r := range results {
		if r.Correct {
			correct++
		}
		if r.Category != "" {
			categoryTotal[r.Category]++
			if r.Correct {
				categoryCorrect[r.Category]++
			}
		}
	}

	accuracy := 0.0
	if total > 0 {
		accuracy = float64(correct) / float64(total)
	}

	var categoryScores map[string]float64
	if len(categoryTotal) > 0 {
		categoryScores = make(map[string]float64, len(categoryTotal))
		for cat, tot := range categoryTotal {
			if tot > 0 {
				categoryScores[cat] = float64(categoryCorrect[cat]) / float64(tot)
			} else {
				categoryScores[cat] = 0.0
			}
		}
	}

	return BenchmarkResult{
		BenchmarkName:   name,
		Accuracy:        accuracy,
		TotalQuestions:  total,
		CorrectCount:    correct,
		TimeSeconds:     timeSeconds,
		QuestionResults: results,
		CategoryScores:  categoryScores,
		ThinkingUsed:    thinkingUsed,
	}
}
