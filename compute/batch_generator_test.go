// SPDX-License-Identifier: MIT OR Apache-2.0

package compute

import (
	"errors"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
	"github.com/tamnd/fastmlx/pipeline"
)

// fakeCache is a per-sequence cache stand-in: an identity object the generator
// hands back through Remove / PromptCache. The real cache holds key/value
// tensors; the bookkeeping only needs object identity.
type fakeCache struct{ id int }

// fakeModel records every Forward call so a test can assert the prefill batch is
// the whole prompt and each decode batch is a single token. Forward returns a
// marker logits row; the chosen token comes from the scripted sampler, so the
// row content is irrelevant to the bookkeeping under test.
type fakeModel struct {
	eos       int
	caches    int
	record    bool      // when true, Forward appends to fedTokens
	fedTokens [][]int32 // one entry per Forward, the tokens fed that step
	forwardEr error     // when set, Forward returns it (the mlx seam stand-in)
}

func (m *fakeModel) NewCache() any {
	m.caches++
	return &fakeCache{id: m.caches}
}

func (m *fakeModel) Forward(tokens []int32, cache any, s *mlxgo.Stream) (any, error) {
	if m.forwardEr != nil {
		return nil, m.forwardEr
	}
	if m.record {
		cp := make([]int32, len(tokens))
		copy(cp, tokens)
		m.fedTokens = append(m.fedTokens, cp)
	}
	return []float32{1, 2, 3}, nil
}

func (m *fakeModel) EOS() int { return m.eos }

// scriptSampler returns a pre-planned token per call, ignoring the logits. It
// records the logits it was handed so a test can confirm processors ran first.
type scriptSampler struct {
	plan []int
	at   int
	seen []any
}

func (s *scriptSampler) Sample(logits any) int {
	s.seen = append(s.seen, logits)
	tok := s.plan[s.at%len(s.plan)]
	s.at++
	return tok
}

// markProc tags the logits row so the sampler can confirm processors ran in
// order before sampling.
type markProc struct{ tag string }

func (p markProc) Apply(logits any) any {
	row, _ := logits.([]float32)
	return append(row, 0) // grow by one so the change is observable
}

func newGen(t *testing.T, m Model) *BatchGenerator {
	t.Helper()
	g, err := NewBatchGenerator(m)
	if err != nil {
		t.Fatalf("NewBatchGenerator: %v", err)
	}
	return g
}

func TestBatchGeneratorPrefillThenDecode(t *testing.T) {
	m := &fakeModel{eos: 999, record: true}
	g := newGen(t, m)
	s := &scriptSampler{plan: []int{100, 101, 102, 103, 104}}

	uid, err := g.Insert(pipeline.DecodeRequest{Tokens: []int{10, 11, 12}, MaxTokens: 5, Sampler: s})
	if err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if uid != 0 {
		t.Fatalf("first uid = %d, want 0", uid)
	}

	var got []int
	var lastFinish string
	for g.HasActive() {
		res, err := g.Step()
		if err != nil {
			t.Fatalf("Step: %v", err)
		}
		if len(res) != 1 {
			t.Fatalf("Step returned %d results, want 1", len(res))
		}
		got = append(got, res[0].Token)
		if res[0].FinishReason != "" {
			lastFinish = res[0].FinishReason
			g.Remove(res[0].UID)
		}
	}

	wantTok := []int{100, 101, 102, 103, 104}
	if len(got) != len(wantTok) {
		t.Fatalf("generated %d tokens, want %d", len(got), len(wantTok))
	}
	for i := range wantTok {
		if got[i] != wantTok[i] {
			t.Fatalf("token[%d] = %d, want %d", i, got[i], wantTok[i])
		}
	}
	if lastFinish != "length" {
		t.Fatalf("finish reason = %q, want length", lastFinish)
	}

	// Prefill fed the whole prompt; each decode fed exactly the prior token.
	if len(m.fedTokens) != 5 {
		t.Fatalf("Forward called %d times, want 5", len(m.fedTokens))
	}
	checkFed(t, m.fedTokens[0], []int32{10, 11, 12})
	for i, prior := range []int32{100, 101, 102, 103} {
		checkFed(t, m.fedTokens[i+1], []int32{prior})
	}
}

func TestBatchGeneratorEOSFinish(t *testing.T) {
	m := &fakeModel{eos: 7}
	g := newGen(t, m)
	s := &scriptSampler{plan: []int{3, 7, 9}} // EOS on the second token

	if _, err := g.Insert(pipeline.DecodeRequest{Tokens: []int{1}, MaxTokens: 100, Sampler: s}); err != nil {
		t.Fatalf("Insert: %v", err)
	}

	r1, _ := g.Step()
	if r1[0].FinishReason != "" {
		t.Fatalf("first token finished early: %q", r1[0].FinishReason)
	}
	r2, _ := g.Step()
	if r2[0].Token != 7 || r2[0].FinishReason != "stop" {
		t.Fatalf("second token = %d/%q, want 7/stop", r2[0].Token, r2[0].FinishReason)
	}
	if r2[0].PromptCache == nil {
		t.Fatalf("finished result missing PromptCache")
	}
}

func TestBatchGeneratorFinishedSeqSkipped(t *testing.T) {
	m := &fakeModel{eos: 5, record: true}
	g := newGen(t, m)
	s := &scriptSampler{plan: []int{5}} // EOS immediately

	g.Insert(pipeline.DecodeRequest{Tokens: []int{1}, MaxTokens: 10, Sampler: s})
	r1, _ := g.Step()
	if r1[0].FinishReason != "stop" {
		t.Fatalf("want stop, got %q", r1[0].FinishReason)
	}
	// A Step before Remove must not emit or advance the finished sequence.
	r2, _ := g.Step()
	if len(r2) != 0 {
		t.Fatalf("finished sequence re-emitted: %+v", r2)
	}
	if len(m.fedTokens) != 1 {
		t.Fatalf("finished sequence forwarded again: %d calls", len(m.fedTokens))
	}
}

func TestBatchGeneratorMultiSequence(t *testing.T) {
	m := &fakeModel{eos: -1}
	g := newGen(t, m)
	uidA, _ := g.Insert(pipeline.DecodeRequest{Tokens: []int{1, 2}, MaxTokens: 2, Sampler: &scriptSampler{plan: []int{20, 21}}})
	uidB, _ := g.Insert(pipeline.DecodeRequest{Tokens: []int{3}, MaxTokens: 2, Sampler: &scriptSampler{plan: []int{30, 31}}})

	res, err := g.Step()
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	if len(res) != 2 {
		t.Fatalf("step emitted %d results, want 2", len(res))
	}
	// Stable, UID-ascending order.
	if res[0].UID != uidA || res[1].UID != uidB {
		t.Fatalf("uids out of order: %d then %d", res[0].UID, res[1].UID)
	}
	if res[0].Token != 20 || res[1].Token != 30 {
		t.Fatalf("first-step tokens = %d,%d want 20,30", res[0].Token, res[1].Token)
	}
}

func TestBatchGeneratorPrefixCacheReused(t *testing.T) {
	m := &fakeModel{eos: -1}
	g := newGen(t, m)
	warm := &fakeCache{id: 42}
	uid, _ := g.Insert(pipeline.DecodeRequest{
		Tokens:    []int{1},
		MaxTokens: 1,
		Sampler:   &scriptSampler{plan: []int{8}},
		Cache:     warm,
	})
	if m.caches != 0 {
		t.Fatalf("NewCache called %d times despite a warm cache", m.caches)
	}
	if got := g.Remove(uid); got != warm {
		t.Fatalf("Remove returned %v, want the warm cache", got)
	}
}

func TestBatchGeneratorRemoveAndClose(t *testing.T) {
	m := &fakeModel{eos: -1}
	g := newGen(t, m)
	uid, _ := g.Insert(pipeline.DecodeRequest{Tokens: []int{1}, MaxTokens: 4, Sampler: &scriptSampler{plan: []int{2}}})
	if !g.HasActive() {
		t.Fatal("HasActive false after Insert")
	}
	if g.Remove(99) != nil {
		t.Fatal("Remove of unknown uid returned non-nil")
	}
	cache := g.Remove(uid)
	if cache == nil {
		t.Fatal("Remove returned nil cache for a live sequence")
	}
	if g.HasActive() {
		t.Fatal("HasActive true after removing the only sequence")
	}
	g.Insert(pipeline.DecodeRequest{Tokens: []int{1}, MaxTokens: 4, Sampler: &scriptSampler{plan: []int{2}}})
	if err := g.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if g.HasActive() {
		t.Fatal("HasActive true after Close")
	}
}

func TestBatchGeneratorEmptyPrompt(t *testing.T) {
	g := newGen(t, &fakeModel{eos: -1})
	if _, err := g.Insert(pipeline.DecodeRequest{Tokens: nil, MaxTokens: 4}); !errors.Is(err, ErrEmptyPrompt) {
		t.Fatalf("Insert empty prompt err = %v, want ErrEmptyPrompt", err)
	}
}

func TestBatchGeneratorLogitsProcsApplied(t *testing.T) {
	m := &fakeModel{eos: -1}
	g := newGen(t, m)
	s := &scriptSampler{plan: []int{1}}
	g.Insert(pipeline.DecodeRequest{
		Tokens:      []int{1},
		MaxTokens:   1,
		Sampler:     s,
		LogitsProcs: []pipeline.LogitsProc{markProc{"a"}, markProc{"b"}},
	})
	if _, err := g.Step(); err != nil {
		t.Fatalf("Step: %v", err)
	}
	// The sampler must have seen the row after both processors grew it.
	row, ok := s.seen[0].([]float32)
	if !ok {
		t.Fatalf("sampler saw %T, want []float32", s.seen[0])
	}
	if len(row) != 5 { // started at 3, +1 per processor
		t.Fatalf("processed row len = %d, want 5", len(row))
	}
}

func TestBatchGeneratorForwardErrorSurfaces(t *testing.T) {
	// The mlx seam: on the default stub the model's Forward returns
	// ErrMLXUnavailable, and Step surfaces it after the host-side gather.
	m := &fakeModel{eos: -1, forwardEr: mlxgo.ErrMLXUnavailable}
	g := newGen(t, m)
	g.Insert(pipeline.DecodeRequest{Tokens: []int{1, 2}, MaxTokens: 4, Sampler: &scriptSampler{plan: []int{0}}})
	_, err := g.Step()
	if !errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("Step err = %v, want ErrMLXUnavailable", err)
	}
	if g.HasActive() != true {
		t.Fatal("sequence dropped on a forward error; it should remain for retry")
	}
}

func checkFed(t *testing.T, got, want []int32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("fed %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("fed %v, want %v", got, want)
		}
	}
}

func BenchmarkBatchGeneratorStep(b *testing.B) {
	m := &fakeModel{eos: -1}
	g, _ := NewBatchGenerator(m)
	const batch = 8
	for range batch {
		g.Insert(pipeline.DecodeRequest{
			Tokens:    []int{1, 2, 3, 4},
			MaxTokens: 0, // no length cap, EOS never hit: steps run forever
			Sampler:   &scriptSampler{plan: []int{5}},
		})
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := g.Step(); err != nil {
			b.Fatalf("Step: %v", err)
		}
	}
}
