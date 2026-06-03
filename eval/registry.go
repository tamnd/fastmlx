// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import "sort"

// This file is the benchmark registry: the name-to-benchmark table the CLI
// resolves a requested benchmark through, mirroring the reference BENCHMARKS
// map. The returned benchmarks are zero-value structs; the few-shot pools the
// stateful ones carry and the sandbox runners the code ones inject are filled
// in by the loader at run time.

// Benchmarks returns a fresh table of every registered benchmark keyed by its
// stable name. A new map is returned on each call so callers can populate the
// per-benchmark seams without sharing state.
func Benchmarks() map[string]Benchmark {
	return map[string]Benchmark{
		"mmlu":          MMLU{},
		"mmlu_pro":      MMLUPro{},
		"kmmlu":         KMMLU{},
		"cmmlu":         CMMLU{},
		"jmmlu":         JMMLU{},
		"hellaswag":     HellaSwag{},
		"truthfulqa":    TruthfulQA{},
		"arc_challenge": ARCChallenge{},
		"winogrande":    Winogrande{},
		"gsm8k":         GSM8K{},
		"mathqa":        MathQA{},
		"humaneval":     HumanEval{},
		"mbpp":          MBPP{},
		"livecodebench": LiveCodeBench{},
		"bbq":           BBQ{},
		"safetybench":   SafetyBench{},
	}
}

// GetBenchmark resolves a benchmark by name, reporting false when no benchmark
// is registered under that name.
func GetBenchmark(name string) (Benchmark, bool) {
	b, ok := Benchmarks()[name]
	return b, ok
}

// BenchmarkNames returns every registered benchmark name in sorted order.
func BenchmarkNames() []string {
	all := Benchmarks()
	names := make([]string, 0, len(all))
	for name := range all {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
