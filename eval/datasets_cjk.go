// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"strconv"
	"strings"
)

// This file holds the Chinese, Japanese, and Korean MMLU-family benchmarks. They
// share a subject-name formatter that treats both dashes and underscores as word
// breaks and the A-D question layout, and differ in their prompt language and in
// whether they prepend same-subject few-shot examples.

var cjkSubjectReplacer = strings.NewReplacer("_", " ", "-", " ")

// formatSubjectNameDashUnderscore turns a subject slug into a readable name,
// treating dashes and underscores as spaces before title-casing.
func formatSubjectNameDashUnderscore(subject string) string {
	return pyTitle(cjkSubjectReplacer.Replace(subject))
}

// CMMLU is the Chinese MMLU: 5-shot multiple choice across Chinese subjects.
// FewShot is the per-subject example pool the loader builds.
type CMMLU struct {
	FewShot map[string][]Item
}

func (CMMLU) Name() string   { return "cmmlu" }
func (CMMLU) QuickSize() int { return 300 }
func (CMMLU) MaxTokens() int { return 128 }

func (c CMMLU) FormatPrompt(item Item) []Message {
	subject := itemStr(item, "subject")
	parts := []string{
		"以下是关于" + formatSubjectNameDashUnderscore(subject) + "的单选题，请直接回答正确选项的字母（A、B、C或D）。\n",
	}
	for _, ex := range c.FewShot[subject] {
		parts = append(parts, mmluFormatQuestion(ex))
		parts = append(parts, "答案: "+itemStr(ex, "answer")+"\n")
	}
	parts = append(parts, mmluFormatQuestion(item))
	parts = append(parts, "答案:")
	return userMessage(strings.Join(parts, "\n"))
}

func (CMMLU) ExtractAnswer(response string, item Item) string {
	return ExtractMCAnswer(response, []string{"A", "B", "C", "D"})
}

func (CMMLU) CheckAnswer(predicted string, item Item) bool {
	return predicted == itemStr(item, "answer")
}

func (CMMLU) Category(item Item) string { return itemStr(item, "subject") }

// BuildCMMLUFewShot collects up to five dev examples per subject. CMMLU answers
// are already letters, defaulting to "A" when absent.
func BuildCMMLUFewShot(devItems []Item) map[string][]Item {
	out := map[string][]Item{}
	for _, raw := range devItems {
		subject := itemStrOr(raw, "subject", "unknown")
		if len(out[subject]) < 5 {
			out[subject] = append(out[subject], Item{
				"question": itemStr(raw, "question"),
				"choices":  itemList(raw, "choices"),
				"answer":   itemStrOr(raw, "answer", "A"),
			})
		}
	}
	return out
}

// JMMLU is the Japanese MMLU: 0-shot multiple choice across Japanese subjects.
type JMMLU struct{}

func (JMMLU) Name() string   { return "jmmlu" }
func (JMMLU) QuickSize() int { return 300 }
func (JMMLU) MaxTokens() int { return 128 }

func (JMMLU) FormatPrompt(item Item) []Message {
	parts := []string{
		"以下は" + formatSubjectNameDashUnderscore(itemStr(item, "subject")) +
			"に関する選択問題です。正解のアルファベット（A、B、C、D）だけを答えてください。\n",
		mmluFormatQuestion(item),
		"答え:",
	}
	return userMessage(strings.Join(parts, "\n"))
}

func (JMMLU) ExtractAnswer(response string, item Item) string {
	return ExtractMCAnswer(response, []string{"A", "B", "C", "D"})
}

func (JMMLU) CheckAnswer(predicted string, item Item) bool {
	return predicted == itemStr(item, "answer")
}

func (JMMLU) Category(item Item) string { return itemStr(item, "subject") }

// kmmluAnswerLetter maps KMMLU's one-based answer index to its letter, falling
// back to the decimal index outside the 1-4 range.
func kmmluAnswerLetter(idx int) string {
	if idx >= 1 && idx <= len(AnswerMap) {
		return AnswerMap[idx-1]
	}
	return strconv.Itoa(idx)
}

// NormalizeKMMLUItem turns a raw KMMLU record into the loader's normalized
// shape, mapping the one-based answer index to a letter.
func NormalizeKMMLUItem(raw Item) Item {
	return Item{
		"question": itemStr(raw, "question"),
		"choices":  itemList(raw, "choices"),
		"answer":   kmmluAnswerLetter(itemInt(raw, "answer")),
		"subject":  itemStrOr(raw, "subject", "unknown"),
	}
}

// KMMLU is the Korean MMLU: 5-shot multiple choice across Korean subjects, with
// answers stored as one-based indices in the raw data.
type KMMLU struct {
	FewShot map[string][]Item
}

func (KMMLU) Name() string   { return "kmmlu" }
func (KMMLU) QuickSize() int { return 300 }
func (KMMLU) MaxTokens() int { return 128 }

func (k KMMLU) FormatPrompt(item Item) []Message {
	subject := itemStr(item, "subject")
	parts := []string{
		"다음은 " + formatSubjectNameDashUnderscore(subject) + "에 대한 객관식 문제입니다. 정답의 알파벳(A, B, C, D)만 답하세요.\n",
	}
	for _, ex := range k.FewShot[subject] {
		parts = append(parts, mmluFormatQuestion(ex))
		parts = append(parts, "정답: "+itemStr(ex, "answer")+"\n")
	}
	parts = append(parts, mmluFormatQuestion(item))
	parts = append(parts, "정답:")
	return userMessage(strings.Join(parts, "\n"))
}

func (KMMLU) ExtractAnswer(response string, item Item) string {
	return ExtractMCAnswer(response, []string{"A", "B", "C", "D"})
}

func (KMMLU) CheckAnswer(predicted string, item Item) bool {
	return predicted == itemStr(item, "answer")
}

func (KMMLU) Category(item Item) string { return itemStr(item, "subject") }

// BuildKMMLUFewShot collects up to five dev examples per subject, mapping each
// one-based answer index to its letter.
func BuildKMMLUFewShot(devItems []Item) map[string][]Item {
	out := map[string][]Item{}
	for _, raw := range devItems {
		subject := itemStrOr(raw, "subject", "unknown")
		if len(out[subject]) < 5 {
			out[subject] = append(out[subject], Item{
				"question": itemStr(raw, "question"),
				"choices":  itemList(raw, "choices"),
				"answer":   kmmluAnswerLetter(itemInt(raw, "answer")),
			})
		}
	}
	return out
}
