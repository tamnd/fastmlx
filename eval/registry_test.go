// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type registryFixture struct {
	Names       []string `json:"names"`
	NamesSorted []string `json:"names_sorted"`
	Count       int      `json:"count"`
}

func loadRegistry(t *testing.T) registryFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/registry.json")
	if err != nil {
		t.Fatal(err)
	}
	var f registryFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestBenchmarkNamesParity(t *testing.T) {
	f := loadRegistry(t)
	if got := BenchmarkNames(); !reflect.DeepEqual(got, f.NamesSorted) {
		t.Errorf("BenchmarkNames() = %v, want %v", got, f.NamesSorted)
	}
	if got := len(Benchmarks()); got != f.Count {
		t.Errorf("registry has %d benchmarks, want %d", got, f.Count)
	}
}

func TestRegistryKeyMatchesName(t *testing.T) {
	for name, b := range Benchmarks() {
		if b.Name() != name {
			t.Errorf("benchmark under key %q reports Name() %q", name, b.Name())
		}
	}
}

func TestGetBenchmark(t *testing.T) {
	for _, name := range loadRegistry(t).Names {
		b, ok := GetBenchmark(name)
		if !ok {
			t.Errorf("GetBenchmark(%q) not found", name)
			continue
		}
		if b.Name() != name {
			t.Errorf("GetBenchmark(%q).Name() = %q", name, b.Name())
		}
	}
	if _, ok := GetBenchmark("does_not_exist"); ok {
		t.Error("GetBenchmark returned ok for an unknown name")
	}
}

func TestBenchmarksFreshMap(t *testing.T) {
	a := Benchmarks()
	delete(a, "mmlu")
	if _, ok := Benchmarks()["mmlu"]; !ok {
		t.Error("Benchmarks() should return a fresh map each call")
	}
}

func BenchmarkBenchmarkNames(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = BenchmarkNames()
	}
}
