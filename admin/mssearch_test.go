// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type msSearchFixture struct {
	Corpus   []map[string]any `json:"corpus"`
	Enriched []map[string]any `json:"enriched"`
	Search   []struct {
		Query string         `json:"query"`
		Sort  string         `json:"sort"`
		Limit int            `json:"limit"`
		Out   map[string]any `json:"out"`
	} `json:"search"`
	Prefilter []struct {
		MaxMemoryBytes int              `json:"max_memory_bytes"`
		ResultLimit    int              `json:"result_limit"`
		Out            []map[string]any `json:"out"`
	} `json:"prefilter"`
	Split []struct {
		MaxMemoryBytes int            `json:"max_memory_bytes"`
		ResultLimit    int            `json:"result_limit"`
		Out            map[string]any `json:"out"`
	} `json:"split"`
}

func loadMSSearch(t *testing.T) msSearchFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/mssearch.json")
	if err != nil {
		t.Fatal(err)
	}
	var f msSearchFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestMSSearchModelsParity(t *testing.T) {
	fx := loadMSSearch(t)
	for i, c := range fx.Search {
		got := jsonRoundTripAny(t, MSSearchModels(fx.Corpus, c.Query, c.Sort, c.Limit))
		want := jsonRoundTripAny(t, c.Out)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("MSSearchModels case %d (q=%q sort=%s):\n got  %v\n want %v", i, c.Query, c.Sort, got, want)
		}
	}
}

func TestMSRecommendedPreFilterParity(t *testing.T) {
	fx := loadMSSearch(t)
	for i, c := range fx.Prefilter {
		got := jsonRoundTripAny(t, MSRecommendedPreFilter(fx.Corpus, c.MaxMemoryBytes, c.ResultLimit))
		want := jsonRoundTripAny(t, c.Out)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("MSRecommendedPreFilter case %d:\n got  %v\n want %v", i, got, want)
		}
	}
}

func TestMSRecommendedSplitParity(t *testing.T) {
	fx := loadMSSearch(t)
	for i, c := range fx.Split {
		got := jsonRoundTripAny(t, MSRecommendedSplit(fx.Enriched, c.MaxMemoryBytes, c.ResultLimit))
		want := jsonRoundTripAny(t, c.Out)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("MSRecommendedSplit case %d:\n got  %v\n want %v", i, got, want)
		}
	}
}

func BenchmarkMSSearchModels(b *testing.B) {
	data, err := os.ReadFile("testdata/mssearch.json")
	if err != nil {
		b.Fatal(err)
	}
	var fx msSearchFixture
	if err := json.Unmarshal(data, &fx); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = MSSearchModels(fx.Corpus, "", "downloads", 100)
	}
}

func BenchmarkMSRecommendedSplit(b *testing.B) {
	data, err := os.ReadFile("testdata/mssearch.json")
	if err != nil {
		b.Fatal(err)
	}
	var fx msSearchFixture
	if err := json.Unmarshal(data, &fx); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = MSRecommendedSplit(fx.Enriched, 24*1024*1024*1024, 50)
	}
}
