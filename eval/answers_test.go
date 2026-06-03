// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"encoding/json"
	"os"
	"testing"
)

type answersFixture struct {
	MC []struct {
		Resp string `json:"resp"`
		Out  string `json:"out"`
	} `json:"mc"`
	Code []struct {
		Resp string `json:"resp"`
		Out  string `json:"out"`
	} `json:"code"`
	Think []struct {
		In  string `json:"in"`
		Out string `json:"out"`
	} `json:"think"`
	Numeric []struct {
		In  string `json:"in"`
		Out string `json:"out"`
	} `json:"numeric"`
	Normalize []struct {
		In  string `json:"in"`
		Out string `json:"out"`
	} `json:"normalize"`
	Check []struct {
		Pred string `json:"pred"`
		Ans  string `json:"ans"`
		Out  bool   `json:"out"`
	} `json:"check"`
	Subject []struct {
		In  string `json:"in"`
		Out string `json:"out"`
	} `json:"subject"`
	Question []struct {
		In struct {
			Question string `json:"question"`
			Choices  []any  `json:"choices"`
		} `json:"in"`
		Out string `json:"out"`
	} `json:"question"`
	Choices []struct {
		In  json.RawMessage `json:"in"`
		Out []any           `json:"out"`
	} `json:"choices"`
	AnswerMap map[string]string `json:"answer_map"`
}

func loadAnswers(t *testing.T) answersFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/answers.json")
	if err != nil {
		t.Fatal(err)
	}
	var f answersFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// canon re-marshals an arbitrary JSON value so two equal-but-differently-typed
// slices compare regardless of element formatting.
func canon(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestExtractMCAnswerParity(t *testing.T) {
	for _, c := range loadAnswers(t).MC {
		if got := ExtractMCAnswer(c.Resp, []string{"A", "B", "C", "D"}); got != c.Out {
			t.Errorf("ExtractMCAnswer(%q) = %q, want %q", c.Resp, got, c.Out)
		}
	}
}

func TestExtractLastCodeBlockParity(t *testing.T) {
	for _, c := range loadAnswers(t).Code {
		if got := ExtractLastCodeBlock(c.Resp); got != c.Out {
			t.Errorf("ExtractLastCodeBlock(%q) = %q, want %q", c.Resp, got, c.Out)
		}
	}
}

func TestStripThinkTagsParity(t *testing.T) {
	for _, c := range loadAnswers(t).Think {
		if got := StripThinkTags(c.In); got != c.Out {
			t.Errorf("StripThinkTags(%q) = %q, want %q", c.In, got, c.Out)
		}
	}
}

func TestExtractNumericAnswerParity(t *testing.T) {
	for _, c := range loadAnswers(t).Numeric {
		if got := ExtractNumericAnswer(c.In); got != c.Out {
			t.Errorf("ExtractNumericAnswer(%q) = %q, want %q", c.In, got, c.Out)
		}
	}
}

func TestNormalizeNumberParity(t *testing.T) {
	for _, c := range loadAnswers(t).Normalize {
		if got := NormalizeNumber(c.In); got != c.Out {
			t.Errorf("NormalizeNumber(%q) = %q, want %q", c.In, got, c.Out)
		}
	}
}

func TestCheckNumericAnswerParity(t *testing.T) {
	for _, c := range loadAnswers(t).Check {
		if got := CheckNumericAnswer(c.Pred, c.Ans); got != c.Out {
			t.Errorf("CheckNumericAnswer(%q, %q) = %v, want %v", c.Pred, c.Ans, got, c.Out)
		}
	}
}

func TestFormatSubjectNameParity(t *testing.T) {
	for _, c := range loadAnswers(t).Subject {
		if got := FormatSubjectName(c.In); got != c.Out {
			t.Errorf("FormatSubjectName(%q) = %q, want %q", c.In, got, c.Out)
		}
	}
}

func TestFormatQuestionParity(t *testing.T) {
	for _, c := range loadAnswers(t).Question {
		if got := FormatQuestion(c.In.Question, c.In.Choices); got != c.Out {
			t.Errorf("FormatQuestion(%q) = %q, want %q", c.In.Question, got, c.Out)
		}
	}
}

func TestParseChoicesParity(t *testing.T) {
	for _, c := range loadAnswers(t).Choices {
		var in any
		if err := json.Unmarshal(c.In, &in); err != nil {
			t.Fatal(err)
		}
		got := ParseChoices(in)
		if canon(t, got) != canon(t, c.Out) {
			t.Errorf("ParseChoices(%s) = %v, want %v", c.In, got, c.Out)
		}
	}
}

func TestAnswerMapParity(t *testing.T) {
	f := loadAnswers(t)
	for i, letter := range AnswerMap {
		key := string(rune('0' + i))
		if f.AnswerMap[key] != letter {
			t.Errorf("AnswerMap[%d] = %q, want %q", i, letter, f.AnswerMap[key])
		}
	}
	if len(f.AnswerMap) != len(AnswerMap) {
		t.Errorf("AnswerMap size = %d, want %d", len(AnswerMap), len(f.AnswerMap))
	}
}

func BenchmarkExtractMCAnswer(b *testing.B) {
	letters := []string{"A", "B", "C", "D"}
	resp := "Let me think... A) wrong B) right. So the final choice is B"
	b.ReportAllocs()
	for b.Loop() {
		_ = ExtractMCAnswer(resp, letters)
	}
}

func BenchmarkExtractNumericAnswer(b *testing.B) {
	resp := "Working through it step by step, we get 12 then 30, so #### 1,234"
	b.ReportAllocs()
	for b.Loop() {
		_ = ExtractNumericAnswer(resp)
	}
}

func BenchmarkNormalizeNumber(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = NormalizeNumber("1,234.0")
	}
}
