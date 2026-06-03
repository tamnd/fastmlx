// SPDX-License-Identifier: MIT OR Apache-2.0

package scheduler

import (
	"encoding/json"
	"os"
	"testing"
)

// progressFixtureResult mirrors a reference get_model_progress entry. The extra
// speculative fields are decoded separately via a raw map so order and presence
// can be checked against extraKeyOrder.
type progressFixtureResult struct {
	RequestID string   `json:"request_id"`
	Processed int      `json:"processed"`
	Total     int      `json:"total"`
	Speed     float64  `json:"speed"`
	ETA       *float64 `json:"eta"`
	Elapsed   float64  `json:"elapsed"`
	Phase     string   `json:"phase"`
	Detail    *string  `json:"detail"`
	raw       map[string]json.RawMessage
}

func (r *progressFixtureResult) UnmarshalJSON(b []byte) error {
	type alias progressFixtureResult
	if err := json.Unmarshal(b, (*alias)(r)); err != nil {
		return err
	}
	return json.Unmarshal(b, &r.raw)
}

// extras returns the present passthrough fields in the fixed reference order.
func (r *progressFixtureResult) extras() []ExtraField {
	var out []ExtraField
	for _, key := range extraKeyOrder {
		if v, ok := r.raw[key]; ok {
			var f float64
			if err := json.Unmarshal(v, &f); err == nil {
				out = append(out, ExtraField{Key: key, Value: f})
			}
		}
	}
	return out
}

type progressFixture struct {
	Events []struct {
		Label   string                  `json:"label"`
		ModelID string                  `json:"model_id"`
		Out     []progressFixtureResult `json:"out"`
	} `json:"events"`
}

// The capture replays update/remove/clear across these monotonic times; the Go
// replay mirrors /tmp/capture_progress.py so each snap() lines up.
func TestPrefillProgressReplay(t *testing.T) {
	raw, err := os.ReadFile("testdata/prefill_progress.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f progressFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}

	tr := NewPrefillProgressTracker()
	scoring := "scoring"
	idx := 0
	want := func(modelID string) {
		ev := f.Events[idx]
		got := tr.GetModelProgress(modelID, snapTimes[idx])
		assertProgress(t, ev.Label, got, ev.Out)
		idx++
	}

	tr.UpdatePrefill("r1", 100, 2048, "qwen3", 10.0)
	want("qwen3")
	tr.UpdatePrefill("r1", 600, 2048, "qwen3", 10.5)
	want("qwen3")
	tr.Update("r2", 50, 400, "gemma", "spec_prefill", &scoring,
		map[string]float64{"scored_tokens": 400, "selected_tokens": 50, "keep_percent": 12.5, "cached_tokens": 0}, 11.0)
	want("gemma")
	want("qwen3")
	tr.Update("r1", 700, 2048, "qwen3", "spec_prefill", nil, nil, 12.0)
	want("qwen3")
	tr.Update("r1", 700, 2048, "qwen3", "spec_prefill", nil, nil, 12.0)
	want("qwen3")
	tr.UpdatePrefill("r1", 2048, 2048, "qwen3", 13.0)
	want("qwen3")
	tr.Remove("r2")
	want("gemma")
	want("nope")
	tr.UpdatePrefill("ra", 10, 100, "qwen3", 14.0)
	tr.UpdatePrefill("rb", 20, 100, "qwen3", 14.0)
	want("qwen3")
	tr.UpdatePrefill("ra", 100, 100, "qwen3", 14.0)
	tr.UpdatePrefill("ra", 30, 100, "qwen3", 14.0)
	want("qwen3")
	tr.UpdatePrefill("r3", 10, 100, "qwen3", 15.0)
	tr.Clear()
	want("qwen3")
}

// snapTimes are the monotonic times at which each snap() was taken in the
// capture (one per event, in order).
var snapTimes = []float64{
	10.0, 10.5, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 13.0, 14.0, 14.0, 15.0,
}

func assertProgress(t *testing.T, label string, got []ProgressResult, want []progressFixtureResult) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d results, want %d", label, len(got), len(want))
	}
	for i := range want {
		w, g := want[i], got[i]
		if g.RequestID != w.RequestID || g.Processed != w.Processed || g.Total != w.Total ||
			g.Speed != w.Speed || g.Elapsed != w.Elapsed || g.Phase != w.Phase {
			t.Errorf("%s[%d]: got %+v, want %+v", label, i, g, w)
		}
		if !eqFloatPtr(g.ETA, w.ETA) {
			t.Errorf("%s[%d] eta: got %v, want %v", label, i, deref(g.ETA), deref(w.ETA))
		}
		if !eqStrPtr(g.Detail, w.Detail) {
			t.Errorf("%s[%d] detail: got %v, want %v", label, i, g.Detail, w.Detail)
		}
		we := w.extras()
		if len(g.Extra) != len(we) {
			t.Fatalf("%s[%d] extras: got %v, want %v", label, i, g.Extra, we)
		}
		for j := range we {
			if g.Extra[j] != we[j] {
				t.Errorf("%s[%d] extra[%d]: got %+v, want %+v", label, i, j, g.Extra[j], we[j])
			}
		}
	}
}

func eqFloatPtr(a, b *float64) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

func eqStrPtr(a, b *string) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

func deref(p *float64) any {
	if p == nil {
		return nil
	}
	return *p
}

func TestGetPrefillTrackerSingleton(t *testing.T) {
	a := GetPrefillTracker()
	b := GetPrefillTracker()
	if a != b {
		t.Error("GetPrefillTracker returned different instances")
	}
}

func BenchmarkPrefillProgressUpdate(b *testing.B) {
	tr := NewPrefillProgressTracker()
	b.ReportAllocs()
	n := 0
	for b.Loop() {
		n++
		tr.UpdatePrefill("r1", n%2000, 2048, "qwen3", float64(n)*0.001)
		_ = tr.GetModelProgress("qwen3", float64(n)*0.001)
	}
}
