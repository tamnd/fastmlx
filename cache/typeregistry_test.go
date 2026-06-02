// SPDX-License-Identifier: MIT OR Apache-2.0

package cache

import (
	"encoding/json"
	"os"
	"testing"
)

type typeRegistryFixture struct {
	Enum              map[string]string `json:"enum"`
	ClassNameMap      map[string]string `json:"class_name_map"`
	RegisteredSlicing map[string]bool   `json:"registered_slicing"`
	DefaultSlicing    bool              `json:"default_slicing"`
	Resolve           []struct {
		Name      string  `json:"name"`
		CacheType *string `json:"cache_type"`
		Sliceable bool    `json:"sliceable"`
	} `json:"resolve"`
	Reverse map[string]string `json:"reverse"`
}

func loadTypeRegistryFixture(t *testing.T) typeRegistryFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/typeregistry.json")
	if err != nil {
		t.Fatal(err)
	}
	var f typeRegistryFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestCacheClassNameMapParity(t *testing.T) {
	fx := loadTypeRegistryFixture(t)
	if len(cacheClassNameMap) != len(fx.ClassNameMap) {
		t.Fatalf("map size %d, want %d", len(cacheClassNameMap), len(fx.ClassNameMap))
	}
	for name, want := range fx.ClassNameMap {
		got, ok := cacheClassNameMap[name]
		if !ok || string(got) != want {
			t.Errorf("class map[%q] = (%q,%v), want %q", name, got, ok, want)
		}
	}
}

func TestRegisteredSlicingParity(t *testing.T) {
	fx := loadTypeRegistryFixture(t)
	if DefaultSlicing != fx.DefaultSlicing {
		t.Errorf("DefaultSlicing = %v, want %v", DefaultSlicing, fx.DefaultSlicing)
	}
	for ctVal, want := range fx.RegisteredSlicing {
		if got := registeredSlicing[CacheType(ctVal)]; got != want {
			t.Errorf("registeredSlicing[%q] = %v, want %v", ctVal, got, want)
		}
	}
	if len(registeredSlicing) != len(fx.RegisteredSlicing) {
		t.Errorf("registeredSlicing size %d, want %d", len(registeredSlicing), len(fx.RegisteredSlicing))
	}
}

func TestResolveAndSliceabilityParity(t *testing.T) {
	fx := loadTypeRegistryFixture(t)
	for _, c := range fx.Resolve {
		ct, ok := CacheTypeForClassName(c.Name)
		if c.CacheType == nil {
			if ok {
				t.Errorf("CacheTypeForClassName(%q) = %q, want unknown", c.Name, ct)
			}
		} else if !ok || string(ct) != *c.CacheType {
			t.Errorf("CacheTypeForClassName(%q) = (%q,%v), want %q", c.Name, ct, ok, *c.CacheType)
		}
		if got := IsSliceableByClassName(c.Name); got != c.Sliceable {
			t.Errorf("IsSliceableByClassName(%q) = %v, want %v", c.Name, got, c.Sliceable)
		}
	}
}

func TestClassNameForTypeParity(t *testing.T) {
	fx := loadTypeRegistryFixture(t)
	for ctVal, want := range fx.Reverse {
		if got := ClassNameForType(CacheType(ctVal)); got != want {
			t.Errorf("ClassNameForType(%q) = %q, want %q", ctVal, got, want)
		}
	}
}

func TestKnownClassNamesOrder(t *testing.T) {
	names := KnownClassNames()
	if len(names) != len(cacheClassNames) {
		t.Fatalf("KnownClassNames len %d, want %d", len(names), len(cacheClassNames))
	}
	for i, e := range cacheClassNames {
		if names[i] != e.name {
			t.Errorf("KnownClassNames[%d] = %q, want %q", i, names[i], e.name)
		}
	}
}

func BenchmarkIsSliceableByClassName(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = IsSliceableByClassName("BatchRotatingKVCache")
	}
}
