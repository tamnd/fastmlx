// SPDX-License-Identifier: MIT OR Apache-2.0

package engine

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"
)

// ocOutput mirrors a reference RequestOutput row in the fixture (the merge
// fields only; the Go RequestOutput carries the same set plus CacheCreation).
type ocOutput struct {
	RequestID        string `json:"request_id"`
	NewTokenIDs      []int  `json:"new_token_ids"`
	NewText          string `json:"new_text"`
	OutputTokenIDs   []int  `json:"output_token_ids"`
	OutputText       string `json:"output_text"`
	Finished         bool   `json:"finished"`
	FinishReason     string `json:"finish_reason"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
	CachedTokens     int    `json:"cached_tokens"`
	Error            string `json:"error"`
}

type ocFixture struct {
	Merge struct {
		AggMerged  *ocOutput `json:"agg_merged"`
		AggAfter   *ocOutput `json:"agg_after"`
		AggErr     *ocOutput `json:"agg_err"`
		Noagg      *ocOutput `json:"noagg"`
		AfterClear *ocOutput `json:"after_clear"`
	} `json:"merge"`
	ShouldSend []struct {
		In  [4]json.RawMessage `json:"in"`
		Out bool               `json:"out"`
	} `json:"should_send"`
	MarkSentSeq []struct {
		Total     int  `json:"total"`
		Finished  bool `json:"finished"`
		Send      bool `json:"send"`
		SentAfter int  `json:"sent_after"`
	} `json:"mark_sent_seq"`
}

func loadOCFixture(t *testing.T) ocFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/output_collector.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f ocFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return f
}

func ro(id string, toks []int, text string, outToks []int, outText string, finished bool, reason, errStr string) RequestOutput {
	return RequestOutput{
		RequestID: id, NewTokenIDs: toks, NewText: text,
		OutputTokenIDs: outToks, OutputText: outText, Finished: finished,
		FinishReason: reason, PromptTokens: 7, CompletionTokens: len(outToks),
		CachedTokens: 2, Err: errStr,
	}
}

func assertOutput(t *testing.T, label string, got RequestOutput, ok bool, want *ocOutput) {
	t.Helper()
	if want == nil {
		if ok {
			t.Errorf("%s: expected no output, got %+v", label, got)
		}
		return
	}
	if !ok {
		t.Fatalf("%s: expected an output, got none", label)
	}
	w := RequestOutput{
		RequestID: want.RequestID, NewTokenIDs: want.NewTokenIDs, NewText: want.NewText,
		OutputTokenIDs: want.OutputTokenIDs, OutputText: want.OutputText, Finished: want.Finished,
		FinishReason: want.FinishReason, PromptTokens: want.PromptTokens,
		CompletionTokens: want.CompletionTokens, CachedTokens: want.CachedTokens, Err: want.Error,
	}
	gj, _ := json.Marshal(got)
	wj, _ := json.Marshal(w)
	if string(gj) != string(wj) {
		t.Errorf("%s:\n got %s\nwant %s", label, gj, wj)
	}
}

func TestCollectorMerge(t *testing.T) {
	f := loadOCFixture(t)

	c := NewRequestOutputCollector(true)
	c.Put(ro("req1", []int{10}, "He", []int{10}, "He", false, "", ""))
	c.Put(ro("req1", []int{11, 12}, "llo", []int{10, 11, 12}, "Hello", false, "", ""))
	out, ok := c.GetNowait()
	assertOutput(t, "agg_merged", out, ok, f.Merge.AggMerged)
	out, ok = c.GetNowait()
	assertOutput(t, "agg_after", out, ok, f.Merge.AggAfter)

	c2 := NewRequestOutputCollector(true)
	c2.Put(ro("req2", []int{1}, "a", []int{1}, "a", false, "", "boom"))
	c2.Put(ro("req2", []int{2}, "b", []int{1, 2}, "ab", false, "", ""))
	out, ok = c2.GetNowait()
	assertOutput(t, "agg_err", out, ok, f.Merge.AggErr)

	c3 := NewRequestOutputCollector(false)
	c3.Put(ro("req3", []int{1}, "a", []int{1}, "a", false, "", ""))
	c3.Put(ro("req3", []int{2}, "b", []int{1, 2}, "ab", true, "stop", ""))
	out, ok = c3.GetNowait()
	assertOutput(t, "noagg", out, ok, f.Merge.Noagg)

	c4 := NewRequestOutputCollector(true)
	c4.Put(ro("req4", []int{9}, "z", []int{9}, "z", false, "", ""))
	c4.Clear()
	out, ok = c4.GetNowait()
	assertOutput(t, "after_clear", out, ok, f.Merge.AfterClear)
}

func TestRequestStreamStateShouldSend(t *testing.T) {
	f := loadOCFixture(t)
	for i, tc := range f.ShouldSend {
		var interval, sent, total int
		var finished bool
		mustInt(t, tc.In[0], &interval)
		mustInt(t, tc.In[1], &sent)
		mustInt(t, tc.In[2], &total)
		mustBool(t, tc.In[3], &finished)
		s := RequestStreamState{StreamInterval: interval, SentTokens: sent}
		if got := s.ShouldSend(total, finished); got != tc.Out {
			t.Errorf("should_send[%d] in=%v: got %v, want %v", i, tc.In, got, tc.Out)
		}
	}
}

func TestRequestStreamStateSequence(t *testing.T) {
	f := loadOCFixture(t)
	s := NewRequestStreamState(3)
	for i, step := range f.MarkSentSeq {
		send := s.ShouldSend(step.Total, step.Finished)
		if send != step.Send {
			t.Errorf("step %d should_send: got %v, want %v", i, send, step.Send)
		}
		if send {
			s.MarkSent(step.Total)
		}
		if s.SentTokens != step.SentAfter {
			t.Errorf("step %d sent_after: got %d, want %d", i, s.SentTokens, step.SentAfter)
		}
	}
}

func TestCollectorBlockingGet(t *testing.T) {
	c := NewRequestOutputCollector(true)
	done := make(chan RequestOutput, 1)
	go func() {
		out, err := c.Get(context.Background())
		if err != nil {
			t.Errorf("Get: %v", err)
		}
		done <- out
	}()
	// Give the consumer a moment to block, then produce.
	time.Sleep(20 * time.Millisecond)
	if !HasWaitingConsumers() {
		t.Error("expected a waiting consumer while Get blocks")
	}
	c.Put(ro("req", []int{1}, "x", []int{1}, "x", false, "", ""))
	select {
	case out := <-done:
		if out.NewText != "x" {
			t.Errorf("got %q, want x", out.NewText)
		}
	case <-time.After(time.Second):
		t.Fatal("Get did not return after Put")
	}
	if HasWaitingConsumers() {
		t.Error("waiting consumer count should be zero after Get returns")
	}
}

func TestCollectorGetContextCancel(t *testing.T) {
	c := NewRequestOutputCollector(true)
	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() {
		_, err := c.Get(ctx)
		errc <- err
	}()
	time.Sleep(20 * time.Millisecond)
	cancel()
	select {
	case err := <-errc:
		if err != context.Canceled {
			t.Errorf("got %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Get did not return after cancel")
	}
	if HasWaitingConsumers() {
		t.Error("waiting consumer count should be zero after cancel")
	}
}

func mustInt(t *testing.T, raw json.RawMessage, dst *int) {
	t.Helper()
	if err := json.Unmarshal(raw, dst); err != nil {
		t.Fatalf("decode int %s: %v", raw, err)
	}
}

func mustBool(t *testing.T, raw json.RawMessage, dst *bool) {
	t.Helper()
	if err := json.Unmarshal(raw, dst); err != nil {
		t.Fatalf("decode bool %s: %v", raw, err)
	}
}

func BenchmarkCollectorPutGet(b *testing.B) {
	c := NewRequestOutputCollector(true)
	out := ro("req", []int{1}, "x", []int{1}, "x", false, "", "")
	b.ReportAllocs()
	for b.Loop() {
		c.Put(out)
		c.Put(out)
		_, _ = c.GetNowait()
	}
}
