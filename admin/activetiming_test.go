// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type timingFixture struct {
	Generating []map[string]any `json:"generating"`
	Waiting    []map[string]any `json:"waiting"`
	Loading    []map[string]any `json:"loading"`
	IdleTTL    []map[string]any `json:"idlettl"`
}

func loadTiming(t *testing.T) timingFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/timing.json")
	if err != nil {
		t.Fatal(err)
	}
	var f timingFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestBuildGeneratingEntryParity(t *testing.T) {
	cases := []struct {
		now          float64
		id           string
		started      any
		lastActivity any
		generated    int
		prompt       int
		maxTokens    any
	}{
		{100, "r1", 90.0, 95.0, 20, 8, 64},
		{100, "r2", nil, nil, 0, 0, nil},
		{100, "r3", 100.0, nil, 5, 3, nil},
		{100, "r4", 97.0, nil, 10, 1, nil},
		{100, "r5", 0.0, nil, 5, 0, nil},
	}
	want := loadTiming(t).Generating
	if len(cases) != len(want) {
		t.Fatalf("case count %d != fixture count %d", len(cases), len(want))
	}
	for i, c := range cases {
		got := jsonRoundTrip(t, BuildGeneratingEntry(c.now, c.id, c.started, c.lastActivity, c.generated, c.prompt, c.maxTokens))
		if !reflect.DeepEqual(got, want[i]) {
			t.Errorf("BuildGeneratingEntry case %d (%s):\n got  %v\n want %v", i, c.id, got, want[i])
		}
	}
}

func TestBuildWaitingEntryParity(t *testing.T) {
	cases := []struct {
		now     float64
		id      string
		pos     int
		arrival float64
		prompt  int
	}{
		{100, "w1", 1, 95.0, 12},
		{100, "w2", 2, 110.0, 0},
	}
	want := loadTiming(t).Waiting
	for i, c := range cases {
		got := jsonRoundTrip(t, BuildWaitingEntry(c.now, c.id, c.pos, c.arrival, c.prompt))
		if !reflect.DeepEqual(got, want[i]) {
			t.Errorf("BuildWaitingEntry case %d (%s):\n got  %v\n want %v", i, c.id, got, want[i])
		}
	}
}

func TestLoadingEstimateParity(t *testing.T) {
	const gb = 1 << 30
	cases := []struct {
		now     float64
		started any
		size    int
		spg     any
		obs     int
	}{
		{100, 90.0, 8 * gb, 2.0, 3},
		{100, nil, 0, 2.0, 3},
		{100, 90.0, 0, 2.0, 1},
		{100, 90.0, 1 * gb, 1.0, 2},
		{100, 88.0, 10 * gb, 1.0, 5},
		{100, 99.0, 4 * gb, 2.0, 2},
	}
	want := loadTiming(t).Loading
	if len(cases) != len(want) {
		t.Fatalf("case count %d != fixture count %d", len(cases), len(want))
	}
	for i, c := range cases {
		got := jsonRoundTrip(t, LoadingEstimate(c.now, c.started, c.size, c.spg, c.obs))
		if !reflect.DeepEqual(got, want[i]) {
			t.Errorf("LoadingEstimate case %d:\n got  %v\n want %v", i, got, want[i])
		}
	}
}

func TestIdleAndTTLParity(t *testing.T) {
	cases := []struct {
		now      float64
		isLoaded bool
		last     any
		ttl      any
	}{
		{1000, true, 900.0, 300},
		{1000, true, nil, 300},
		{1000, false, 900.0, 300},
		{1000, true, 0.0, 300},
		{1000, true, 900.0, 50},
		{1000, true, 1100.0, 300},
	}
	want := loadTiming(t).IdleTTL
	if len(cases) != len(want) {
		t.Fatalf("case count %d != fixture count %d", len(cases), len(want))
	}
	for i, c := range cases {
		got := jsonRoundTrip(t, IdleAndTTL(c.now, c.isLoaded, c.last, c.ttl))
		if !reflect.DeepEqual(got, want[i]) {
			t.Errorf("IdleAndTTL case %d:\n got  %v\n want %v", i, got, want[i])
		}
	}
}

func BenchmarkBuildGeneratingEntry(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = BuildGeneratingEntry(100, "r1", 90.0, 95.0, 20, 8, 64)
	}
}

func BenchmarkLoadingEstimate(b *testing.B) {
	const gb = 1 << 30
	b.ReportAllocs()
	for b.Loop() {
		_ = LoadingEstimate(100, 90.0, 8*gb, 2.0, 3)
	}
}
