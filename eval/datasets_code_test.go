// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"encoding/json"
	"os"
	"testing"
)

type heScriptCase struct {
	Code       string `json:"code"`
	Test       string `json:"test"`
	EntryPoint string `json:"entry_point"`
	Out        string `json:"out"`
}

type mbppScriptCase struct {
	Code     string   `json:"code"`
	TestList []string `json:"test_list"`
	Setup    string   `json:"setup"`
	Out      string   `json:"out"`
}

type humanEvalFixture struct {
	Normalize []normalizeCase `json:"normalize"`
	Prompt    []promptCase    `json:"prompt"`
	Extract   []extractCase   `json:"extract"`
	Script    []heScriptCase  `json:"script"`
	Check     []checkCase     `json:"check"`
	Category  []categoryCase  `json:"category"`
}

type mbppFixture struct {
	Normalize []normalizeCase  `json:"normalize"`
	Prompt    []promptCase     `json:"prompt"`
	Extract   []extractCase    `json:"extract"`
	Script    []mbppScriptCase `json:"script"`
	Check     []checkCase      `json:"check"`
	Category  []categoryCase   `json:"category"`
}

type lcbNormalizeCase struct {
	Raw map[string]any `json:"raw"`
	Idx int            `json:"idx"`
	Out map[string]any `json:"out"`
}

type lcbFixture struct {
	Normalize []lcbNormalizeCase `json:"normalize"`
	Prompt    []promptCase       `json:"prompt"`
	Extract   []extractCase      `json:"extract"`
	Check     []checkCase        `json:"check"`
	Category  []categoryCase     `json:"category"`
}

type codeFixture struct {
	HumanEval     humanEvalFixture `json:"humaneval"`
	MBPP          mbppFixture      `json:"mbpp"`
	LiveCodeBench lcbFixture       `json:"livecodebench"`
}

func loadCode(t *testing.T) codeFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/datasets_code.json")
	if err != nil {
		t.Fatal(err)
	}
	var f codeFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// recordingRunner is a fake CodeRunner: it captures the last script it was
// handed and returns a canned verdict, so CheckAnswer's wiring can be checked
// without executing anything.
type recordingRunner struct {
	passed bool
	script string
}

func (r *recordingRunner) Run(script string) (bool, string) {
	r.script = script
	return r.passed, ""
}

func TestHumanEvalParity(t *testing.T) {
	f := loadCode(t).HumanEval
	checkBenchmark(t, HumanEval{}, benchFixture{
		Prompt:   f.Prompt,
		Extract:  f.Extract,
		Check:    f.Check,
		Category: f.Category,
	})
}

func TestHumanEvalNormalizeParity(t *testing.T) {
	for i, c := range loadCode(t).HumanEval.Normalize {
		got := NormalizeHumanEvalItem(c.Raw)
		if !jsonEqual(t, got, c.Out) {
			gb, _ := json.Marshal(got)
			wb, _ := json.Marshal(c.Out)
			t.Errorf("NormalizeHumanEvalItem case %d = %s, want %s", i, gb, wb)
		}
	}
}

func TestHumanEvalScriptParity(t *testing.T) {
	for i, c := range loadCode(t).HumanEval.Script {
		if got := HumanEvalScript(c.Code, c.Test, c.EntryPoint); got != c.Out {
			t.Errorf("HumanEvalScript case %d =\n%q\nwant\n%q", i, got, c.Out)
		}
	}
}

func TestHumanEvalCheckRunner(t *testing.T) {
	item := Item{"test": "def check(c):\n    pass\n", "entry_point": "add"}
	runner := &recordingRunner{passed: true}
	h := HumanEval{Runner: runner}
	if !h.CheckAnswer("def add(a, b):\n    return a + b", item) {
		t.Fatal("expected pass when runner returns true")
	}
	want := HumanEvalScript("def add(a, b):\n    return a + b", "def check(c):\n    pass\n", "add")
	if runner.script != want {
		t.Errorf("runner got script\n%q\nwant\n%q", runner.script, want)
	}
	runner.passed = false
	if (HumanEval{Runner: runner}).CheckAnswer("def add(): pass", item) {
		t.Error("expected fail when runner returns false")
	}
	if (HumanEval{Runner: runner}).CheckAnswer("   ", item) {
		t.Error("expected fail for blank prediction before invoking runner")
	}
}

func TestMBPPParity(t *testing.T) {
	f := loadCode(t).MBPP
	checkBenchmark(t, MBPP{}, benchFixture{
		Prompt:   f.Prompt,
		Extract:  f.Extract,
		Check:    f.Check,
		Category: f.Category,
	})
}

func TestMBPPNormalizeParity(t *testing.T) {
	for i, c := range loadCode(t).MBPP.Normalize {
		got, ok := NormalizeMBPPItem(c.Raw)
		// The reference skips records without a test list; the fixture encodes
		// that as a null normalized output.
		if c.Out == nil {
			if ok {
				t.Errorf("NormalizeMBPPItem case %d: expected skip, got %v", i, got)
			}
			continue
		}
		if !ok {
			t.Errorf("NormalizeMBPPItem case %d: unexpected skip", i)
			continue
		}
		if !jsonEqual(t, got, c.Out) {
			gb, _ := json.Marshal(got)
			wb, _ := json.Marshal(c.Out)
			t.Errorf("NormalizeMBPPItem case %d = %s, want %s", i, gb, wb)
		}
	}
}

func TestMBPPScriptParity(t *testing.T) {
	for i, c := range loadCode(t).MBPP.Script {
		if got := MBPPScript(c.Code, c.TestList, c.Setup); got != c.Out {
			t.Errorf("MBPPScript case %d =\n%q\nwant\n%q", i, got, c.Out)
		}
	}
}

func TestMBPPCheckRunner(t *testing.T) {
	item := Item{
		"test_list":       []any{"assert f() == 1", "assert f() != 2"},
		"test_setup_code": "import math",
	}
	runner := &recordingRunner{passed: true}
	m := MBPP{Runner: runner}
	if !m.CheckAnswer("def f():\n    return 1", item) {
		t.Fatal("expected pass when runner returns true")
	}
	want := MBPPScript("def f():\n    return 1", []string{"assert f() == 1", "assert f() != 2"}, "import math")
	if runner.script != want {
		t.Errorf("runner got script\n%q\nwant\n%q", runner.script, want)
	}
	if (MBPP{Runner: runner}).CheckAnswer("", item) {
		t.Error("expected fail for blank prediction before invoking runner")
	}
}

// stdinScript pairs the code and stdin a StdinRunner was handed.
type stdinScript struct {
	code  string
	stdin string
}

// scriptedStdin is a fake StdinRunner: it returns a canned stdout/success for
// each call in order and records the (code, stdin) pairs it received.
type scriptedStdin struct {
	results []struct {
		stdout  string
		success bool
	}
	calls []stdinScript
	n     int
}

func (s *scriptedStdin) Run(code, stdin string) (string, bool, string) {
	s.calls = append(s.calls, stdinScript{code, stdin})
	r := s.results[s.n]
	s.n++
	return r.stdout, r.success, ""
}

func TestLiveCodeBenchParity(t *testing.T) {
	f := loadCode(t).LiveCodeBench
	checkBenchmark(t, LiveCodeBench{}, benchFixture{
		Prompt:   f.Prompt,
		Extract:  f.Extract,
		Check:    f.Check,
		Category: f.Category,
	})
}

func TestLiveCodeBenchNormalizeParity(t *testing.T) {
	for i, c := range loadCode(t).LiveCodeBench.Normalize {
		got, ok := NormalizeLiveCodeBenchItem(c.Raw, c.Idx)
		if c.Out == nil {
			if ok {
				t.Errorf("NormalizeLiveCodeBenchItem case %d: expected skip, got %v", i, got)
			}
			continue
		}
		if !ok {
			t.Errorf("NormalizeLiveCodeBenchItem case %d: unexpected skip", i)
			continue
		}
		if !jsonEqual(t, got, c.Out) {
			gb, _ := json.Marshal(got)
			wb, _ := json.Marshal(c.Out)
			t.Errorf("NormalizeLiveCodeBenchItem case %d = %s, want %s", i, gb, wb)
		}
	}
}

func TestLiveCodeBenchCheckRunner(t *testing.T) {
	item := Item{
		"inputs":  []any{"1 2\n", "4 5\n", "7 8\n", "9 9\n"},
		"outputs": []any{"3\n", " 9 ", "15", "18"},
	}
	// All three of the first three test cases pass (only the first three run);
	// stdout is compared after trimming, so " 9 " matches "9".
	runner := &scriptedStdin{results: []struct {
		stdout  string
		success bool
	}{{"3", true}, {"9", true}, {"15", true}}}
	l := LiveCodeBench{Runner: runner}
	if !l.CheckAnswer("print(1)", item) {
		t.Fatal("expected pass when all of the first three cases match")
	}
	if len(runner.calls) != 3 {
		t.Errorf("expected only the first 3 cases to run, got %d", len(runner.calls))
	}
	if runner.calls[0].stdin != "1 2\n" {
		t.Errorf("first stdin = %q, want %q", runner.calls[0].stdin, "1 2\n")
	}

	// A mismatched stdout fails the run.
	mismatch := &scriptedStdin{results: []struct {
		stdout  string
		success bool
	}{{"3", true}, {"wrong", true}, {"15", true}}}
	if (LiveCodeBench{Runner: mismatch}).CheckAnswer("print(1)", item) {
		t.Error("expected fail on stdout mismatch")
	}

	// A non-zero exit fails the run immediately.
	crash := &scriptedStdin{results: []struct {
		stdout  string
		success bool
	}{{"", false}, {"9", true}, {"15", true}}}
	if (LiveCodeBench{Runner: crash}).CheckAnswer("print(1)", item) {
		t.Error("expected fail when a case exits non-zero")
	}

	// Blank prediction never invokes the runner.
	if (LiveCodeBench{Runner: runner}).CheckAnswer("  ", item) {
		t.Error("expected fail for blank prediction")
	}
}

func BenchmarkLiveCodeBenchFormatPrompt(b *testing.B) {
	item := Item{"description": "Read two integers and print their sum."}
	bench := LiveCodeBench{}
	b.ReportAllocs()
	for b.Loop() {
		_ = bench.FormatPrompt(item)
	}
}

func BenchmarkHumanEvalFormatPrompt(b *testing.B) {
	item := Item{"prompt": "def add(a, b):\n    \"\"\"Add.\"\"\"\n"}
	bench := HumanEval{}
	b.ReportAllocs()
	for b.Loop() {
		_ = bench.FormatPrompt(item)
	}
}

func BenchmarkMBPPFormatPrompt(b *testing.B) {
	item := Item{
		"prompt":    "Write a function to add two numbers.",
		"test_list": []any{"assert add(1,2)==3", "assert add(0,0)==0", "assert add(-1,1)==0", "assert add(5,5)==10"},
	}
	bench := MBPP{}
	b.ReportAllocs()
	for b.Loop() {
		_ = bench.FormatPrompt(item)
	}
}
