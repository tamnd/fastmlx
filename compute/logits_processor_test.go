// SPDX-License-Identifier: MIT OR Apache-2.0

package compute

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

type lpParams struct {
	LogitBias             map[string]float64 `json:"logit_bias"`
	RepetitionPenalty     *float64           `json:"repetition_penalty"`
	RepetitionContextSize int                `json:"repetition_context_size"`
	PresencePenalty       *float64           `json:"presence_penalty"`
	PresenceContextSize   int                `json:"presence_context_size"`
	FrequencyPenalty      *float64           `json:"frequency_penalty"`
	FrequencyContextSize  int                `json:"frequency_context_size"`
}

func (p lpParams) toParams(t *testing.T) LogitsProcessorParams {
	var bias map[int]float64
	if len(p.LogitBias) > 0 {
		bias = make(map[int]float64, len(p.LogitBias))
		for k, v := range p.LogitBias {
			id, err := strconv.Atoi(k)
			if err != nil {
				t.Fatalf("bad logit_bias key %q: %v", k, err)
			}
			bias[id] = v
		}
	}
	return LogitsProcessorParams{
		LogitBias:             bias,
		RepetitionPenalty:     p.RepetitionPenalty,
		RepetitionContextSize: p.RepetitionContextSize,
		PresencePenalty:       p.PresencePenalty,
		PresenceContextSize:   p.PresenceContextSize,
		FrequencyPenalty:      p.FrequencyPenalty,
		FrequencyContextSize:  p.FrequencyContextSize,
	}
}

func loadLPCases(t *testing.T) []struct {
	Label    string    `json:"label"`
	Logits   []float64 `json:"logits"`
	Tokens   []int     `json:"tokens"`
	Params   lpParams  `json:"params"`
	Expected []float64 `json:"expected"`
} {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "logits_processors.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Label    string    `json:"label"`
		Logits   []float64 `json:"logits"`
		Tokens   []int     `json:"tokens"`
		Params   lpParams  `json:"params"`
		Expected []float64 `json:"expected"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	return cases
}

func TestLogitsProcessorsParity(t *testing.T) {
	for _, c := range loadLPCases(t) {
		logits := make([]float32, len(c.Logits))
		for i, v := range c.Logits {
			logits[i] = float32(v)
		}
		procs := MakeLogitsProcessors(c.Params.toParams(t))
		ApplyLogitsProcessors(procs, c.Tokens, logits)

		if len(logits) != len(c.Expected) {
			t.Fatalf("%s: len = %d, want %d", c.Label, len(logits), len(c.Expected))
		}
		for i, want := range c.Expected {
			if logits[i] != float32(want) {
				t.Errorf("%s: logits[%d] = %v, want %v", c.Label, i, logits[i], float32(want))
			}
		}
	}
}

func TestRepetitionPenaltyNoCompound(t *testing.T) {
	// A token repeated in the window is penalized exactly once, from its
	// original value, not iteratively. Three 4.0s would compound to 4/1.5/1.5
	// if applied per occurrence; the mlx gather/scatter applies it once.
	logits := []float32{4.0, -2.0}
	MakeRepetitionPenalty(1.5, 20)([]int{0, 0, 0}, logits)
	if want := float32(4.0) / float32(1.5); logits[0] != want {
		t.Errorf("logits[0] = %v, want %v (single application)", logits[0], want)
	}
}

func TestPenaltyOutOfRangeTokensSkipped(t *testing.T) {
	// Out-of-range token ids (defensive guard) must not panic and must leave
	// the row untouched.
	logits := []float32{1.0, 2.0}
	MakePresencePenalty(0.5, 20)([]int{5, -3}, logits)
	MakeFrequencyPenalty(0.5, 20)([]int{99}, logits)
	if logits[0] != 1.0 || logits[1] != 2.0 {
		t.Errorf("row mutated by out-of-range tokens: %v", logits)
	}
}

func TestMakeLogitsProcessorsSelection(t *testing.T) {
	zero := 0.0
	one := 1.3
	// nil and zero penalties are dropped; non-zero ones are kept, in order.
	params := LogitsProcessorParams{
		RepetitionPenalty: &one,
		PresencePenalty:   &zero, // dropped: == 0
		FrequencyPenalty:  nil,   // dropped: nil
	}
	if got := len(MakeLogitsProcessors(params)); got != 1 {
		t.Fatalf("processor count = %d, want 1", got)
	}
	// empty logit_bias adds nothing
	params.LogitBias = map[int]float64{}
	if got := len(MakeLogitsProcessors(params)); got != 1 {
		t.Fatalf("processor count with empty bias = %d, want 1", got)
	}
}

func BenchmarkLogitsProcessors(b *testing.B) {
	b.ReportAllocs()
	rep, pres, freq := 1.3, 0.4, 0.2
	params := LogitsProcessorParams{
		LogitBias:         map[int]float64{0: 1.5, 3: -2.0},
		RepetitionPenalty: &rep,
		PresencePenalty:   &pres,
		FrequencyPenalty:  &freq,
	}
	procs := MakeLogitsProcessors(params)
	tokens := []int{3, 3, 1, 5, 3, 2, 7, 4, 6, 0}
	base := make([]float32, 4096)
	for i := range base {
		base[i] = float32(i%17) - 8
	}
	logits := make([]float32, len(base))
	for b.Loop() {
		copy(logits, base)
		ApplyLogitsProcessors(procs, tokens, logits)
	}
}
