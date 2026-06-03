// SPDX-License-Identifier: MIT OR Apache-2.0

package compute

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type samplerFixture struct {
	Cases []struct {
		Label  string                 `json:"label"`
		Logits []float64              `json:"logits"`
		Op     string                 `json:"op"`
		Params map[string]json.Number `json:"params"`
		Kept   []bool                 `json:"kept"`
	} `json:"cases"`
	Argmax struct {
		Logits []float64 `json:"logits"`
		Index  int       `json:"index"`
	} `json:"argmax"`
}

func loadSamplerFixture(t *testing.T) samplerFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "sampler_filters.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx samplerFixture
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&fx); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	return fx
}

func TestSamplerFiltersParity(t *testing.T) {
	fx := loadSamplerFixture(t)
	for _, c := range fx.Cases {
		in := make([]float32, len(c.Logits))
		for i, v := range c.Logits {
			in[i] = float32(v)
		}
		// Filters operate on log-probs (logits - logsumexp), as make_sampler feeds them.
		logits := logSoftmax(in)
		switch c.Op {
		case "apply_top_p":
			ApplyTopP(logits, paramF(t, c.Params, "top_p"))
		case "apply_min_p":
			ApplyMinP(logits, paramF(t, c.Params, "min_p"), paramI(c.Params, "min_tokens_to_keep"))
		case "apply_top_k":
			ApplyTopK(logits, paramI(c.Params, "top_k"))
		default:
			t.Fatalf("%s: unknown op %q", c.Label, c.Op)
		}
		if len(logits) != len(c.Kept) {
			t.Fatalf("%s: len %d, want %d", c.Label, len(logits), len(c.Kept))
		}
		for i := range logits {
			masked := math.IsInf(float64(logits[i]), -1)
			if c.Kept[i] == masked {
				t.Errorf("%s: token %d kept=%v but masked=%v", c.Label, i, c.Kept[i], masked)
			}
		}
	}
}

func TestArgmaxParity(t *testing.T) {
	fx := loadSamplerFixture(t)
	in := make([]float32, len(fx.Argmax.Logits))
	for i, v := range fx.Argmax.Logits {
		in[i] = float32(v)
	}
	if got := Argmax(in); got != fx.Argmax.Index {
		t.Errorf("Argmax = %d, want %d", got, fx.Argmax.Index)
	}
	if Argmax(nil) != -1 {
		t.Error("Argmax(nil) should be -1")
	}
}

func TestSamplerComposition(t *testing.T) {
	// Greedy: temp 0 leaves the row untouched; caller uses Argmax.
	s := Sampler{Temp: 0, TopP: 0.5, TopK: 2}
	if !s.Greedy() {
		t.Fatal("temp 0 should be greedy")
	}
	row := []float32{1, 2, 3}
	cp := append([]float32(nil), row...)
	s.ApplyFilters(row)
	for i := range row {
		if row[i] != cp[i] {
			t.Errorf("greedy mutated row at %d: %v", i, row[i])
		}
	}
	// Non-greedy with top_k 1 keeps only the max.
	s = Sampler{Temp: 1.0, TopK: 1}
	row = []float32{1, 5, 3}
	s.ApplyFilters(row)
	if math.IsInf(float64(row[1]), -1) {
		t.Error("top_k 1 masked the max")
	}
	if !math.IsInf(float64(row[0]), -1) || !math.IsInf(float64(row[2]), -1) {
		t.Errorf("top_k 1 kept a non-max: %v", row)
	}
}

func BenchmarkSamplerFilters(b *testing.B) {
	b.ReportAllocs()
	s := Sampler{Temp: 1.0, TopP: 0.9, MinP: 0.05, TopK: 64, MinTokensToKeep: 1}
	base := make([]float32, 4096)
	for i := range base {
		base[i] = float32(math.Sin(float64(i)) * 4)
	}
	logits := make([]float32, len(base))
	for b.Loop() {
		copy(logits, base)
		s.ApplyFilters(logits)
	}
}

// logSoftmax returns logits - logsumexp(logits), the log-prob row the sampler
// filters consume, computed in float32 to match the capture's mx input.
func logSoftmax(in []float32) []float32 {
	out := make([]float32, len(in))
	var maxv float32 = negInf
	for _, v := range in {
		if v > maxv {
			maxv = v
		}
	}
	var sum float64
	for _, v := range in {
		sum += math.Exp(float64(v - maxv))
	}
	lse := float64(maxv) + math.Log(sum)
	for i, v := range in {
		out[i] = float32(float64(v) - lse)
	}
	return out
}

func paramF(t *testing.T, p map[string]json.Number, key string) float64 {
	v, ok := p[key]
	if !ok {
		t.Fatalf("missing param %q", key)
	}
	f, err := v.Float64()
	if err != nil {
		t.Fatalf("param %q: %v", key, err)
	}
	return f
}

func paramI(p map[string]json.Number, key string) int {
	v, ok := p[key]
	if !ok {
		return 1
	}
	i, err := v.Int64()
	if err != nil {
		return 1
	}
	return int(i)
}
