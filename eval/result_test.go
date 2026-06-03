// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

// stubMC is a minimal A-to-D multiple-choice benchmark used to drive
// ScoreQuestion: it scores by exact match and reads its category from the
// "subject" field, matching the capture stub.
type stubMC struct{}

func (stubMC) Name() string                     { return "stub" }
func (stubMC) QuickSize() int                   { return 0 }
func (stubMC) MaxTokens() int                   { return 0 }
func (stubMC) FormatPrompt(item Item) []Message { return nil }
func (stubMC) Category(item Item) string        { return itemStr(item, "subject") }
func (stubMC) CheckAnswer(pred string, it Item) bool {
	return pred == itemStr(it, "answer")
}
func (stubMC) ExtractAnswer(response string, item Item) string {
	return ExtractMCAnswer(response, []string{"A", "B", "C", "D"})
}

type scoreCase struct {
	Item        map[string]any `json:"item"`
	Idx         int            `json:"idx"`
	Response    string         `json:"response"`
	PromptText  string         `json:"prompt_text"`
	TimeSeconds float64        `json:"time_seconds"`
	Out         QuestionResult `json:"out"`
}

type aggregateCase struct {
	Name         string           `json:"name"`
	TimeSeconds  float64          `json:"time_seconds"`
	ThinkingUsed bool             `json:"thinking_used"`
	Results      []QuestionResult `json:"results"`
	Out          BenchmarkResult  `json:"out"`
}

type resultFixture struct {
	Score     []scoreCase     `json:"score"`
	Aggregate []aggregateCase `json:"aggregate"`
}

func loadResult(t *testing.T) resultFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/result.json")
	if err != nil {
		t.Fatal(err)
	}
	var f resultFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestScoreQuestionParity(t *testing.T) {
	for i, c := range loadResult(t).Score {
		got := ScoreQuestion(stubMC{}, c.Item, c.Idx, c.Response, c.PromptText, c.TimeSeconds)
		if !reflect.DeepEqual(got, c.Out) {
			t.Errorf("ScoreQuestion case %d =\n%+v\nwant\n%+v", i, got, c.Out)
		}
	}
}

func TestAggregateParity(t *testing.T) {
	for i, c := range loadResult(t).Aggregate {
		got := Aggregate(c.Name, c.Results, c.TimeSeconds, c.ThinkingUsed)
		if !reflect.DeepEqual(got, c.Out) {
			t.Errorf("Aggregate case %d =\n%+v\nwant\n%+v", i, got, c.Out)
		}
	}
}

func BenchmarkAggregate(b *testing.B) {
	results := make([]QuestionResult, 100)
	for i := range results {
		cat := "physics"
		if i%2 == 0 {
			cat = "law"
		}
		results[i] = QuestionResult{QuestionID: asStr(i), Correct: i%3 == 0, Category: cat}
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = Aggregate("mmlu", results, 1.0, false)
	}
}
