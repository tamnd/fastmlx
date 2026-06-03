// SPDX-License-Identifier: MIT OR Apache-2.0

package engine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type requestFixture struct {
	FinishReason map[string]*string `json:"finish_reason"`
	IsFinished   map[string]bool    `json:"is_finished"`
	Less         []struct {
		APriority int     `json:"a_priority"`
		AArrival  float64 `json:"a_arrival"`
		BPriority int     `json:"b_priority"`
		BArrival  float64 `json:"b_arrival"`
		Less      bool    `json:"less"`
	} `json:"less"`
	Usage []struct {
		PromptTokens     int            `json:"prompt_tokens"`
		CompletionTokens int            `json:"completion_tokens"`
		Usage            map[string]int `json:"usage"`
	} `json:"usage"`
}

// statusByName maps the reference enum names to the fastmlx statuses. The Go
// enum carries two extra members (StatusPrefilling, StatusFinishedError) so the
// integer values differ; the mapping is by meaning.
var statusByName = map[string]RequestStatus{
	"WAITING":                StatusWaiting,
	"RUNNING":                StatusRunning,
	"PREEMPTED":              StatusPreempted,
	"FINISHED_STOPPED":       StatusFinishedStopped,
	"FINISHED_LENGTH_CAPPED": StatusFinishedLength,
	"FINISHED_ABORTED":       StatusFinishedAborted,
}

func loadRequestFixture(t *testing.T) requestFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "request.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx requestFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	return fx
}

func arrivalTime(seconds float64) time.Time {
	base := time.Unix(0, 0).UTC()
	return base.Add(time.Duration(seconds * float64(time.Second)))
}

func TestStatusGetFinishReason(t *testing.T) {
	fx := loadRequestFixture(t)
	for name, want := range fx.FinishReason {
		st, ok := statusByName[name]
		if !ok {
			t.Fatalf("unmapped status name %q", name)
		}
		got := st.GetFinishReason()
		wantStr := ""
		if want != nil {
			wantStr = *want
		}
		if got != wantStr {
			t.Errorf("%s GetFinishReason = %q, want %q", name, got, wantStr)
		}
	}
}

func TestStatusIsFinished(t *testing.T) {
	fx := loadRequestFixture(t)
	for name, want := range fx.IsFinished {
		st := statusByName[name]
		if got := st.Finished(); got != want {
			t.Errorf("%s Finished = %v, want %v", name, got, want)
		}
	}
}

func TestRequestLess(t *testing.T) {
	fx := loadRequestFixture(t)
	for i, tc := range fx.Less {
		a := &Request{Priority: tc.APriority, Arrival: arrivalTime(tc.AArrival)}
		b := &Request{Priority: tc.BPriority, Arrival: arrivalTime(tc.BArrival)}
		if got := a.Less(b); got != tc.Less {
			t.Errorf("case %d: Less = %v, want %v", i, got, tc.Less)
		}
	}
}

func TestRequestOutputUsage(t *testing.T) {
	fx := loadRequestFixture(t)
	for _, tc := range fx.Usage {
		o := &RequestOutput{PromptTokens: tc.PromptTokens, CompletionTokens: tc.CompletionTokens}
		got := o.Usage()
		for k, want := range tc.Usage {
			if got[k] != want {
				t.Errorf("usage[%q] = %d, want %d (prompt=%d completion=%d)", k, got[k], want, tc.PromptTokens, tc.CompletionTokens)
			}
		}
		if len(got) != len(tc.Usage) {
			t.Errorf("usage has %d keys, want %d", len(got), len(tc.Usage))
		}
	}
}

func TestRequestGetFinishReasonExplicitWins(t *testing.T) {
	r := &Request{Status: StatusFinishedStopped, FinishReason: "custom"}
	if got := r.GetFinishReason(); got != "custom" {
		t.Errorf("explicit reason = %q, want custom", got)
	}
	r2 := &Request{Status: StatusFinishedLength}
	if got := r2.GetFinishReason(); got != "length" {
		t.Errorf("derived reason = %q, want length", got)
	}
}

func TestRequestSetFinished(t *testing.T) {
	r := &Request{}
	r.SetFinished(StatusFinishedAborted, "")
	if r.Status != StatusFinishedAborted || r.FinishReason != "abort" {
		t.Errorf("derived: status=%v reason=%q", r.Status, r.FinishReason)
	}
	r.SetFinished(StatusFinishedStopped, "manual")
	if r.FinishReason != "manual" {
		t.Errorf("explicit reason = %q, want manual", r.FinishReason)
	}
}

func BenchmarkRequestLess(b *testing.B) {
	a := &Request{Priority: 0, Arrival: arrivalTime(1.0)}
	other := &Request{Priority: 0, Arrival: arrivalTime(2.0)}
	b.ReportAllocs()
	for b.Loop() {
		_ = a.Less(other)
	}
}
