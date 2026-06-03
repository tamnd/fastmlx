// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import "strings"

// This file holds the code-generation benchmarks HumanEval and MBPP. Their
// prompts, answer extraction, item normalization, and test-script assembly are
// pure and live here. Scoring runs the assembled script in a sandboxed
// subprocess, which is the one compute seam: it is injected as a CodeRunner so
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
