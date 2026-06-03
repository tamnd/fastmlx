// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"encoding/json"
	"os"
	"reflect"
	"strconv"
	"testing"
)

type sampleFixture struct {
	GetRandBits []struct {
		Seed uint64   `json:"seed"`
		Ks   []int    `json:"ks"`
		Out  []uint32 `json:"out"`
	} `json:"getrandbits"`
	RandBelow []struct {
		Seed uint64 `json:"seed"`
		Ns   []int  `json:"ns"`
		Out  []int  `json:"out"`
	} `json:"randbelow"`
	Sample []struct {
		Seed uint64 `json:"seed"`
		N    int    `json:"n"`
		K    int    `json:"k"`
		Out  []int  `json:"out"`
	} `json:"sample"`
	Deterministic []struct {
		N   int      `json:"n"`
		Ids []string `json:"ids"`
	} `json:"deterministic"`
	Stratified []struct {
		Dataset string   `json:"dataset"`
		N       int      `json:"n"`
		Ids     []string `json:"ids"`
	} `json:"stratified"`
}

func loadSample(t *testing.T) sampleFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/sample.json")
	if err != nil {
		t.Fatal(err)
	}
	var f sampleFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

type evalItem struct {
	ID      string
	Subject string
}

// mkItems builds the same synthetic dataset the capture script used: a global
// running index across groups, each item tagged with its subject.
func mkItems(spec [][2]any) []evalItem {
	var items []evalItem
	idx := 0
	for _, g := range spec {
		subj := g[0].(string)
		cnt := g[1].(int)
		for range cnt {
			items = append(items, evalItem{ID: strconv.Itoa(idx), Subject: subj})
			idx++
		}
	}
	return items
}

func datasets() map[string][]evalItem {
	return map[string][]evalItem{
		"ds1":   mkItems([][2]any{{"alpha", 10}, {"beta", 20}, {"gamma", 5}}),
		"ds2":   mkItems([][2]any{{"math", 100}, {"hist", 50}, {"bio", 30}, {"art", 7}}),
		"small": mkItems([][2]any{{"x", 3}, {"y", 4}}),
	}
}

func TestGetRandBitsParity(t *testing.T) {
	for _, c := range loadSample(t).GetRandBits {
		r := NewPyRandom(c.Seed)
		for i, k := range c.Ks {
			if got := r.GetRandBits(k); got != c.Out[i] {
				t.Errorf("GetRandBits seq[%d] k=%d = %d, want %d", i, k, got, c.Out[i])
			}
		}
	}
}

func TestRandBelowParity(t *testing.T) {
	for _, c := range loadSample(t).RandBelow {
		r := NewPyRandom(c.Seed)
		for i, n := range c.Ns {
			if got := r.RandBelow(n); got != c.Out[i] {
				t.Errorf("RandBelow seq[%d] n=%d = %d, want %d", i, n, got, c.Out[i])
			}
		}
	}
}

func TestSampleIndicesParity(t *testing.T) {
	for _, c := range loadSample(t).Sample {
		r := NewPyRandom(c.Seed)
		got := r.SampleIndices(c.N, c.K)
		if !reflect.DeepEqual(got, c.Out) {
			t.Errorf("SampleIndices(seed=%d, n=%d, k=%d) = %v, want %v", c.Seed, c.N, c.K, got, c.Out)
		}
	}
}

func TestDeterministicSampleParity(t *testing.T) {
	ds1 := datasets()["ds1"]
	for _, c := range loadSample(t).Deterministic {
		got := ids(DeterministicSample(ds1, c.N))
		if !equalStrings(got, c.Ids) {
			t.Errorf("DeterministicSample(n=%d) = %v, want %v", c.N, got, c.Ids)
		}
	}
}

func TestStratifiedSampleParity(t *testing.T) {
	ds := datasets()
	for _, c := range loadSample(t).Stratified {
		got := ids(StratifiedSample(ds[c.Dataset], c.N, func(it evalItem) string { return it.Subject }))
		if !equalStrings(got, c.Ids) {
			t.Errorf("StratifiedSample(%s, n=%d) = %v, want %v", c.Dataset, c.N, got, c.Ids)
		}
	}
}

func ids(items []evalItem) []string {
	out := make([]string, len(items))
	for i, it := range items {
		out[i] = it.ID
	}
	return out
}

// equalStrings treats two empty slices as equal regardless of nil.
func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func BenchmarkSampleIndices(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		r := NewPyRandom(SampleSeed)
		_ = r.SampleIndices(1000, 60)
	}
}

func BenchmarkStratifiedSample(b *testing.B) {
	ds := datasets()["ds2"]
	keyFn := func(it evalItem) string { return it.Subject }
	b.ReportAllocs()
	for b.Loop() {
		_ = StratifiedSample(ds, 30, keyFn)
	}
}
