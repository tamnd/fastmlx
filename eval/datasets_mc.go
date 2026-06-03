// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import "strings"

// This file holds the pure cores of the GSM8K, ARC-Challenge, and HellaSwag
// benchmarks. Each is the format/extract/score logic; the bundled-data load and
// the engine call stay the caller's seams.

// gsm8kFewShot are the standard five chain-of-thought examples prepended to a
// GSM8K prompt.
var gsm8kFewShot = []struct{ question, answer string }{
	{
		"There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
		"There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted. #### 6",
	},
	{
		"If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
		"There are originally 3 cars. Then 2 more arrive. So there are 3 + 2 = 5 cars. #### 5",
	},
	{
		"Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
		"Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39",
	},
	{
		"Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
		"Jason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops. #### 8",
	},
	{
		"Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
		"Shawn started with 5 toys. He got 2 from mom and 2 from dad, so 2 + 2 = 4 more toys. Now he has 5 + 4 = 9 toys. #### 9",
	},
}

// GSM8K is the grade-school-math benchmark: 5-shot chain-of-thought prompting and
// answer extraction from the "#### N" marker.
type GSM8K struct{}

func (GSM8K) Name() string   { return "gsm8k" }
func (GSM8K) QuickSize() int { return 100 }
func (GSM8K) MaxTokens() int { return 512 }

func (GSM8K) FormatPrompt(item Item) []Message {
	parts := []string{
		"Solve the following math problem step by step. " +
			"End your answer with #### followed by the final numeric answer.\n",
	}
	for _, ex := range gsm8kFewShot {
		parts = append(parts, "Question: "+ex.question)
		parts = append(parts, "Answer: "+ex.answer+"\n")
	}
	parts = append(parts, "Question: "+itemStr(item, "question"))
	parts = append(parts, "Answer:")
	return userMessage(strings.Join(parts, "\n"))
}

func (GSM8K) ExtractAnswer(response string, item Item) string {
	return ExtractNumericAnswer(response)
}

func (GSM8K) CheckAnswer(predicted string, item Item) bool {
	return CheckNumericAnswer(predicted, itemStr(item, "answer"))
}

func (GSM8K) Category(item Item) string { return "" }

// ARCChallenge is the ARC-Challenge science-reasoning benchmark: 0-shot multiple
// choice with per-item letter labels.
type ARCChallenge struct{}

func (ARCChallenge) Name() string   { return "arc_challenge" }
func (ARCChallenge) QuickSize() int { return 300 }
func (ARCChallenge) MaxTokens() int { return 128 }

func (ARCChallenge) FormatPrompt(item Item) []Message {
	parts := []string{
		"Answer the following science question. Answer with just the letter.\n",
		"Question: " + itemStr(item, "question") + "\n",
	}
	labels := itemList(item, "labels")
	choices := itemList(item, "choices")
	for i := 0; i < len(labels) && i < len(choices); i++ {
		parts = append(parts, asStr(labels[i])+". "+asStr(choices[i]))
	}
	parts = append(parts, "\nAnswer:")
	return userMessage(strings.Join(parts, "\n"))
}

func (ARCChallenge) ExtractAnswer(response string, item Item) string {
	valid := []string{"A", "B", "C", "D"}
	if labels := itemList(item, "labels"); labels != nil {
		valid = valid[:0]
		for _, l := range labels {
			valid = append(valid, asStr(l))
		}
	}
	return ExtractMCAnswer(response, valid)
}

func (ARCChallenge) CheckAnswer(predicted string, item Item) bool {
	return predicted == itemStr(item, "answer")
}

func (ARCChallenge) Category(item Item) string { return "" }

// HellaSwag is the commonsense-continuation benchmark: 0-shot multiple choice
// over four endings, scored against an integer gold index.
type HellaSwag struct{}

func (HellaSwag) Name() string   { return "hellaswag" }
func (HellaSwag) QuickSize() int { return 200 }
func (HellaSwag) MaxTokens() int { return 128 }

func (HellaSwag) FormatPrompt(item Item) []Message {
	parts := []string{
		"Choose the most plausible continuation. Answer with just the letter (A, B, C, or D).\n",
		"Context: " + itemStr(item, "context") + "\n",
	}
	endings := itemList(item, "endings")
	for i := 0; i < len(endings) && i < 4; i++ {
		parts = append(parts, letterFor(i)+". "+asStr(endings[i]))
	}
	parts = append(parts, "\nAnswer:")
	return userMessage(strings.Join(parts, "\n"))
}

func (HellaSwag) ExtractAnswer(response string, item Item) string {
	return ExtractMCAnswer(response, []string{"A", "B", "C", "D"})
}

func (HellaSwag) CheckAnswer(predicted string, item Item) bool {
	return predicted == letterFor(itemInt(item, "answer"))
}

func (HellaSwag) Category(item Item) string {
	return itemStr(item, "activity_label")
}

// asStr renders a choice value as text, leaving a string untouched and otherwise
// using its default formatting.
func asStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return toStr(v)
}
