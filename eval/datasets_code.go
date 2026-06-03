// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"encoding/json"
	"strings"
)

// This file holds the code-generation benchmarks HumanEval, MBPP, and
// LiveCodeBench. Their prompts, answer extraction, item normalization, and
// test-script assembly are pure and live here. Scoring runs the candidate in a
// sandboxed subprocess, which is the one compute seam: it is injected as a
// CodeRunner (assertion-based) or StdinRunner (stdin/stdout) so
// the script the runner receives stays exact and testable.

// CodeRunner executes an assembled Python script and reports whether it exited
// cleanly along with any captured stderr. The code benchmarks build the script
// text themselves and hand it here, so the sandbox, timeout, and resource
// limits stay on the caller's side.
type CodeRunner interface {
	Run(script string) (passed bool, stderr string)
}

// itemStrList coerces a list-valued item field into a slice of strings.
func itemStrList(item Item, key string) []string {
	raw := itemList(item, key)
	out := make([]string, 0, len(raw))
	for _, v := range raw {
		out = append(out, asStr(v))
	}
	return out
}

// getImports pulls the import lines out of a prompt, preserving each original
// line, mirroring the reference helper of the same purpose.
func getImports(prompt string) string {
	var lines []string
	for line := range strings.SplitSeq(prompt, "\n") {
		stripped := strings.TrimSpace(line)
		if strings.HasPrefix(stripped, "import ") || strings.HasPrefix(stripped, "from ") {
			lines = append(lines, line)
		}
	}
	return strings.Join(lines, "\n")
}

// hasImportLine reports whether any line of code already starts (once trimmed)
// with an import statement.
func hasImportLine(code string) bool {
	for line := range strings.SplitSeq(code, "\n") {
		stripped := strings.TrimSpace(line)
		if strings.HasPrefix(stripped, "import ") || strings.HasPrefix(stripped, "from ") {
			return true
		}
	}
	return false
}

// HumanEval is the function-completion benchmark: the model is handed a
// signature plus docstring and must complete the body, scored by running the
// bundled unit tests. Runner is the injected sandbox seam.
type HumanEval struct {
	Runner CodeRunner
}

func (HumanEval) Name() string   { return "humaneval" }
func (HumanEval) QuickSize() int { return 100 }
func (HumanEval) MaxTokens() int { return 2048 }

// NormalizeHumanEvalItem turns a raw HumanEval record into the loader's shape.
func NormalizeHumanEvalItem(raw Item) Item {
	return Item{
		"id":          itemStr(raw, "task_id"),
		"prompt":      itemStr(raw, "prompt"),
		"test":        itemStr(raw, "test"),
		"entry_point": itemStr(raw, "entry_point"),
		"question":    itemStr(raw, "prompt"),
	}
}

func (HumanEval) FormatPrompt(item Item) []Message {
	content := "Complete the following Python function. " +
		"Provide only the complete function implementation, no explanations.\n\n" +
		itemStr(item, "prompt")
	return userMessage(content)
}

func (HumanEval) ExtractAnswer(response string, item Item) string {
	code := ExtractLastCodeBlock(response)
	imports := getImports(itemStr(item, "prompt"))
	if strings.Contains(code, "def ") && imports != "" {
		if !hasImportLine(code) {
			return imports + "\n\n" + code
		}
	}
	if !strings.Contains(code, "def ") {
		return itemStr(item, "prompt") + code
	}
	return code
}

// HumanEvalScript assembles the verification script: the candidate code, the
// bundled test, and the call into check() against the entry point.
func HumanEvalScript(code, test, entryPoint string) string {
	return code + "\n\n" + test + "\n\n" + "check(" + entryPoint + ")\n"
}

func (h HumanEval) CheckAnswer(predicted string, item Item) bool {
	if strings.TrimSpace(predicted) == "" || h.Runner == nil {
		return false
	}
	passed, _ := h.Runner.Run(HumanEvalScript(predicted, itemStr(item, "test"), itemStr(item, "entry_point")))
	return passed
}

func (HumanEval) Category(item Item) string { return "" }

// MBPP is the Mostly Basic Python Problems benchmark: a natural-language task
// plus assertion tests. Runner is the injected sandbox seam.
type MBPP struct {
	Runner CodeRunner
}

func (MBPP) Name() string   { return "mbpp" }
func (MBPP) QuickSize() int { return 200 }
func (MBPP) MaxTokens() int { return 2048 }

// NormalizeMBPPItem turns a raw MBPP record into the loader's shape, reporting
// false when the record carries no test list and should be skipped.
func NormalizeMBPPItem(raw Item) (Item, bool) {
	testList := itemList(raw, "test_list")
	if len(testList) == 0 {
		return nil, false
	}
	return Item{
		"id":              asStr(raw["task_id"]),
		"prompt":          itemStr(raw, "prompt"),
		"test_list":       testList,
		"test_setup_code": itemStr(raw, "test_setup_code"),
		"question":        itemStr(raw, "prompt"),
	}, true
}

func (MBPP) FormatPrompt(item Item) []Message {
	tests := itemStrList(item, "test_list")
	if len(tests) > 3 {
		tests = tests[:3]
	}
	content := "Write a Python function to solve the following problem. " +
		"Provide only the complete function implementation, no explanations.\n\n" +
		"Problem: " + itemStr(item, "prompt") + "\n\n" +
		"Test cases:\n" + strings.Join(tests, "\n") + "\n\n" +
		"Solution:"
	return userMessage(content)
}

func (MBPP) ExtractAnswer(response string, item Item) string {
	return ExtractLastCodeBlock(response)
}

// MBPPScript assembles the verification script: optional setup code, the
// candidate code, and the joined assertion tests.
func MBPPScript(code string, testList []string, setupCode string) string {
	return setupCode + "\n" + code + "\n" + strings.Join(testList, "\n") + "\n"
}

func (m MBPP) CheckAnswer(predicted string, item Item) bool {
	if strings.TrimSpace(predicted) == "" || m.Runner == nil {
		return false
	}
	passed, _ := m.Runner.Run(MBPPScript(predicted, itemStrList(item, "test_list"), itemStr(item, "test_setup_code")))
	return passed
}

func (MBPP) Category(item Item) string { return "" }

// StdinRunner executes a complete Python program with the given stdin and
// reports its stdout, whether it exited cleanly, and any stderr. It is the
// compute seam LiveCodeBench injects; the input feeding and output comparison
// stay pure on the benchmark's side.
type StdinRunner interface {
	Run(code, stdin string) (stdout string, success bool, stderr string)
}

// LiveCodeBench is the competitive-programming benchmark: the model writes a
// full program that reads stdin and prints stdout, scored by running it against
// the public test cases. Runner is the injected sandbox seam.
type LiveCodeBench struct {
	Runner StdinRunner
}

func (LiveCodeBench) Name() string   { return "livecodebench" }
func (LiveCodeBench) QuickSize() int { return 100 }
func (LiveCodeBench) MaxTokens() int { return 16384 }

// NormalizeLiveCodeBenchItem turns a raw LiveCodeBench record into the loader's
// shape, parsing the public test cases (a JSON string or an already-decoded
// list) into parallel input and output slices. It reports false when there are
// no usable test cases, so the record should be skipped, mirroring the loader's
// per-index fallbacks for the id and title.
func NormalizeLiveCodeBenchItem(raw Item, idx int) (Item, bool) {
	testCases := parseTestCases(raw["public_test_cases"])
	if len(testCases) == 0 {
		return nil, false
	}

	inputs := make([]any, 0, len(testCases))
	outputs := make([]any, 0, len(testCases))
	for _, tc := range testCases {
		m, ok := tc.(map[string]any)
		if !ok {
			inputs = append(inputs, "")
			outputs = append(outputs, "")
			continue
		}
		inputs = append(inputs, mapGet(m, "input"))
		outputs = append(outputs, mapGet(m, "output"))
	}
	if len(inputs) == 0 || len(outputs) == 0 {
		return nil, false
	}

	id := asStr(idx)
	if v, ok := raw["question_id"]; ok {
		id = asStr(v)
	}
	title := "Problem " + asStr(idx)
	if v, ok := raw["question_title"]; ok {
		title = asStr(v)
	}

	return Item{
		"id":           id,
		"title":        title,
		"description":  itemStr(raw, "question_content"),
		"inputs":       inputs,
		"outputs":      outputs,
		"difficulty":   itemStr(raw, "difficulty"),
		"starter_code": itemStr(raw, "starter_code"),
	}, true
}

// parseTestCases coerces the public_test_cases field, which arrives either as a
// JSON-encoded string or an already-decoded list, into a slice. An unparseable
// string or any other shape yields an empty slice.
func parseTestCases(v any) []any {
	switch tc := v.(type) {
	case nil:
		return parseTestCases("[]")
	case []any:
		return tc
	case string:
		var out []any
		if err := json.Unmarshal([]byte(tc), &out); err != nil {
			return nil
		}
		return out
	default:
		return nil
	}
}

// mapGet returns the value at key, or "" when absent.
func mapGet(m map[string]any, key string) any {
	if v, ok := m[key]; ok {
		return v
	}
	return ""
}

func (LiveCodeBench) FormatPrompt(item Item) []Message {
	content := "Solve the following programming problem in Python. " +
		"Read input from stdin and print the output to stdout. " +
		"Provide only the complete Python code, no explanations.\n\n" +
		"Problem:\n" + itemStr(item, "description") + "\n\n" +
		"Solution:"
	return userMessage(content)
}

func (LiveCodeBench) ExtractAnswer(response string, item Item) string {
	return ExtractLastCodeBlock(response)
}

func (l LiveCodeBench) CheckAnswer(predicted string, item Item) bool {
	if strings.TrimSpace(predicted) == "" || l.Runner == nil {
		return false
	}
	inputs := itemStrList(item, "inputs")
	outputs := itemStrList(item, "outputs")
	if len(inputs) > 3 {
		inputs = inputs[:3]
	}
	if len(outputs) > 3 {
		outputs = outputs[:3]
	}
	n := min(len(inputs), len(outputs))
	for i := range n {
		stdout, success, _ := l.Runner.Run(predicted, inputs[i])
		if !success {
			return false
		}
		if strings.TrimSpace(stdout) != strings.TrimSpace(outputs[i]) {
			return false
		}
	}
	return true
}

func (LiveCodeBench) Category(item Item) string { return "" }
