// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"strconv"
	"strings"
)

// This file holds the pure cores of three more benchmarks: MMLU (5-shot,
// per-subject few-shot), Winogrande (2-choice coreference), and TruthfulQA
// (single-correct multiple choice with as many options as the item carries).

// indexToLetter maps a zero-based index to a choice letter, extending past D the
// way TruthfulQA does for items with more than four options.
func indexToLetter(idx int) string {
	return string(rune('A' + idx))
}

// mmluAnswerLetter maps a stored answer index to its letter, falling back to the
// decimal index when it is outside the A-D range, matching the reference's
// `ANSWER_MAP.get(idx, str(idx))`.
func mmluAnswerLetter(idx int) string {
	if l := letterFor(idx); l != "" {
		return l
	}
	return strconv.Itoa(idx)
}

// NormalizeMMLUItem turns a raw MMLU record into the loader's normalized shape:
// the question, the parsed choices, the answer as a letter, and the subject.
func NormalizeMMLUItem(raw Item) Item {
	return Item{
		"question": itemStr(raw, "question"),
		"choices":  ParseChoices(raw["choices"]),
		"answer":   mmluAnswerLetter(itemInt(raw, "answer")),
		"subject":  itemStrOr(raw, "subject", "unknown"),
	}
}

// BuildMMLUFewShot collects up to five normalized dev examples per subject, the
// few-shot pool MMLU prompts draw from.
func BuildMMLUFewShot(devItems []Item) map[string][]Item {
	out := map[string][]Item{}
	for _, raw := range devItems {
		subject := itemStrOr(raw, "subject", "unknown")
		if len(out[subject]) < 5 {
			n := NormalizeMMLUItem(raw)
			// The few-shot pool keeps only the prompt-facing fields, matching
			// the reference, which omits the subject from each example.
			out[subject] = append(out[subject], Item{
				"question": n["question"],
				"choices":  n["choices"],
				"answer":   n["answer"],
			})
		}
	}
	return out
}

// MMLU is the 57-subject knowledge benchmark: 5-shot multiple choice with the
// shots drawn from the same subject. FewShot is the per-subject example pool the
// loader builds; an empty map gives a 0-shot prompt.
type MMLU struct {
	FewShot map[string][]Item
}

func (MMLU) Name() string   { return "mmlu" }
func (MMLU) QuickSize() int { return 300 }
func (MMLU) MaxTokens() int { return 128 }

func (m MMLU) FormatPrompt(item Item) []Message {
	subject := itemStr(item, "subject")
	parts := []string{
		"The following are multiple choice questions about " + FormatSubjectName(subject) +
			". Answer with just the letter (A, B, C, or D).\n",
	}
	for _, ex := range m.FewShot[subject] {
		parts = append(parts, mmluFormatQuestion(ex))
		parts = append(parts, "Answer: "+itemStr(ex, "answer")+"\n")
	}
	parts = append(parts, mmluFormatQuestion(item))
	parts = append(parts, "Answer:")
	return userMessage(strings.Join(parts, "\n"))
}

func (MMLU) ExtractAnswer(response string, item Item) string {
	return ExtractMCAnswer(response, []string{"A", "B", "C", "D"})
}

func (MMLU) CheckAnswer(predicted string, item Item) bool {
	return predicted == itemStr(item, "answer")
}

func (MMLU) Category(item Item) string {
	return itemStr(item, "subject")
}

// mmluFormatQuestion renders a question and its lettered choices, the helper
// MMLU reuses for the shots and the live question.
func mmluFormatQuestion(item Item) string {
	return FormatQuestion(itemStr(item, "question"), itemList(item, "choices"))
}

// Winogrande is the coreference benchmark: pick which of two options fills the
// blank, answered as the digit 1 or 2.
type Winogrande struct{}

func (Winogrande) Name() string   { return "winogrande" }
func (Winogrande) QuickSize() int { return 300 }
func (Winogrande) MaxTokens() int { return 128 }

func (Winogrande) FormatPrompt(item Item) []Message {
	parts := []string{
		"Choose the correct option to fill in the blank (_). Answer with just the number (1 or 2).\n",
		"Sentence: " + itemStr(item, "sentence") + "\n",
		"1. " + itemStr(item, "option1"),
		"2. " + itemStr(item, "option2"),
		"\nAnswer:",
	}
	return userMessage(strings.Join(parts, "\n"))
}

func (Winogrande) ExtractAnswer(response string, item Item) string {
	return ExtractMCAnswer(response, []string{"1", "2"})
}

func (Winogrande) CheckAnswer(predicted string, item Item) bool {
	return predicted == itemStr(item, "answer")
}

func (Winogrande) Category(item Item) string { return "" }

// TruthfulQA is the truthfulness benchmark: 0-shot single-correct multiple
// choice with one lettered option per choice and an integer gold index.
type TruthfulQA struct{}

func (TruthfulQA) Name() string   { return "truthfulqa" }
func (TruthfulQA) QuickSize() int { return 200 }
func (TruthfulQA) MaxTokens() int { return 128 }

func (TruthfulQA) FormatPrompt(item Item) []Message {
	parts := []string{
		"Answer the following question truthfully. Choose the most accurate answer. Answer with just the letter.\n",
		"Question: " + itemStr(item, "question") + "\n",
	}
	for i, choice := range itemList(item, "choices") {
		parts = append(parts, indexToLetter(i)+". "+asStr(choice))
	}
	parts = append(parts, "\nAnswer:")
	return userMessage(strings.Join(parts, "\n"))
}

func (TruthfulQA) ExtractAnswer(response string, item Item) string {
	n := len(itemList(item, "choices"))
	valid := make([]string, n)
	for i := range valid {
		valid[i] = indexToLetter(i)
	}
	return ExtractMCAnswer(response, valid)
}

func (TruthfulQA) CheckAnswer(predicted string, item Item) bool {
	return predicted == indexToLetter(itemInt(item, "answer"))
}

func (TruthfulQA) Category(item Item) string { return "" }

// itemStrOr reads a string field, returning the fallback when it is absent or
// not a string.
func itemStrOr(item Item, key, fallback string) string {
	if v, ok := item[key].(string); ok {
		return v
	}
	return fallback
}
