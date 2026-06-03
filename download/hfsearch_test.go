// SPDX-License-Identifier: MIT OR Apache-2.0

package download

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type hfModelFixture struct {
	ID            string           `json:"id"`
	Downloads     int64            `json:"downloads"`
	Likes         int64            `json:"likes"`
	TrendingScore int64            `json:"trending_score"`
	Parameters    map[string]int64 `json:"parameters"`
}

func (m hfModelFixture) toHubModel() HubModel {
	return HubModel{
		ID:            m.ID,
		Downloads:     m.Downloads,
		Likes:         m.Likes,
		TrendingScore: m.TrendingScore,
		Parameters:    m.Parameters,
	}
}

type hfSearchFixture struct {
	Corpus      []hfModelFixture `json:"corpus"`
	Recommended []struct {
		MaxMemoryBytes int64            `json:"max_memory_bytes"`
		ResultLimit    int              `json:"result_limit"`
		Out            []map[string]any `json:"out"`
	} `json:"recommended"`
	Search []struct {
		Opts struct {
			Sort          string `json:"sort"`
			Limit         int    `json:"limit"`
			MinParams     *int64 `json:"min_params"`
			MaxParams     *int64 `json:"max_params"`
			MinSize       *int64 `json:"min_size"`
			MaxSize       *int64 `json:"max_size"`
			SortBySize    bool   `json:"sort_by_size"`
			SortAscending bool   `json:"sort_ascending"`
		} `json:"opts"`
		Out map[string]any `json:"out"`
	} `json:"search"`
}

// hfRoundTrip decodes a value through JSON so both sides compare with the same
// types (numbers as float64, null as nil), skirting Go vs Python repr gaps.
func hfRoundTrip(t *testing.T, v any) any {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func loadHFSearch(t *testing.T) hfSearchFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/hfsearch.json")
	if err != nil {
		t.Fatal(err)
	}
	var f hfSearchFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func corpusModels(fx hfSearchFixture) []HubModel {
	models := make([]HubModel, len(fx.Corpus))
	for i, m := range fx.Corpus {
		models[i] = m.toHubModel()
	}
	return models
}

func TestRecommendedListParity(t *testing.T) {
	fx := loadHFSearch(t)
	models := corpusModels(fx)
	for i, c := range fx.Recommended {
		got := hfRoundTrip(t, RecommendedList(models, c.MaxMemoryBytes, c.ResultLimit))
		want := hfRoundTrip(t, c.Out)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("RecommendedList case %d:\n got  %v\n want %v", i, got, want)
		}
	}
}

func TestSearchModelsParity(t *testing.T) {
	fx := loadHFSearch(t)
	models := corpusModels(fx)
	for i, c := range fx.Search {
		opts := SearchOptions{
			Sort:          c.Opts.Sort,
			Limit:         c.Opts.Limit,
			MinParams:     c.Opts.MinParams,
			MaxParams:     c.Opts.MaxParams,
			MinSize:       c.Opts.MinSize,
			MaxSize:       c.Opts.MaxSize,
			SortBySize:    c.Opts.SortBySize,
			SortAscending: c.Opts.SortAscending,
		}
		got := hfRoundTrip(t, SearchModels(models, opts))
		want := hfRoundTrip(t, c.Out)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("SearchModels case %d (sort=%s):\n got  %v\n want %v", i, c.Opts.Sort, got, want)
		}
	}
}

func BenchmarkSearchModels(b *testing.B) {
	fx := hfSearchFixture{}
	data, err := os.ReadFile("testdata/hfsearch.json")
	if err != nil {
		b.Fatal(err)
	}
	if err := json.Unmarshal(data, &fx); err != nil {
		b.Fatal(err)
	}
	models := make([]HubModel, len(fx.Corpus))
	for i, m := range fx.Corpus {
		models[i] = m.toHubModel()
	}
	opts := SearchOptions{Sort: "largest", Limit: 100}
	b.ReportAllocs()
	for b.Loop() {
		_ = SearchModels(models, opts)
	}
}

func BenchmarkRecommendedList(b *testing.B) {
	fx := hfSearchFixture{}
	data, err := os.ReadFile("testdata/hfsearch.json")
	if err != nil {
		b.Fatal(err)
	}
	if err := json.Unmarshal(data, &fx); err != nil {
		b.Fatal(err)
	}
	models := make([]HubModel, len(fx.Corpus))
	for i, m := range fx.Corpus {
		models[i] = m.toHubModel()
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = RecommendedList(models, 24*1024*1024*1024, 50)
	}
}
