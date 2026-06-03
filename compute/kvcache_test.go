// SPDX-License-Identifier: MIT OR Apache-2.0

package compute

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type kvRecord struct {
	S        *int  `json:"S"`
	Offset   int   `json:"offset"`
	Capacity int   `json:"capacity"`
	Idx      *int  `json:"idx"`
	Size     *int  `json:"size"`
	Fetch    *int  `json:"fetch"`
	Trim     *int  `json:"trim"`
	Trimmed  *int  `json:"trimmed"`
	Trimmble *bool `json:"trimmable"`
}

type kvFixture struct {
	KV []struct {
		Label   string     `json:"label"`
		Steps   []int      `json:"steps"`
		Trims   []int      `json:"trims"`
		Records []kvRecord `json:"records"`
	} `json:"kv"`
	Rot []struct {
		Label   string     `json:"label"`
		MaxSize int        `json:"max_size"`
		Keep    int        `json:"keep"`
		Steps   []int      `json:"steps"`
		Trims   []int      `json:"trims"`
		Records []kvRecord `json:"records"`
	} `json:"rot"`
}

func loadKVFixture(t *testing.T) kvFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "kvcache_bookkeeping.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx kvFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	return fx
}

func TestKVCacheBookkeepingParity(t *testing.T) {
	fx := loadKVFixture(t)
	for _, sc := range fx.KV {
		c := &KVCache{}
		ri := 0
		for _, s := range sc.Steps {
			plan := c.Update(s)
			r := sc.Records[ri]
			ri++
			if plan.Offset != r.Offset || plan.Capacity != r.Capacity {
				t.Errorf("%s step S=%d: offset/cap = %d/%d, want %d/%d", sc.Label, s, plan.Offset, plan.Capacity, r.Offset, r.Capacity)
			}
			if r.Size != nil && c.Size() != *r.Size {
				t.Errorf("%s step S=%d: size = %d, want %d", sc.Label, s, c.Size(), *r.Size)
			}
			if r.Fetch != nil && plan.FetchEnd != *r.Fetch {
				t.Errorf("%s step S=%d: fetch = %d, want %d", sc.Label, s, plan.FetchEnd, *r.Fetch)
			}
		}
		for _, n := range sc.Trims {
			got := c.Trim(n)
			r := sc.Records[ri]
			ri++
			if r.Trimmed != nil && got != *r.Trimmed {
				t.Errorf("%s trim %d: trimmed = %d, want %d", sc.Label, n, got, *r.Trimmed)
			}
			if c.Offset != r.Offset || c.capacity != r.Capacity {
				t.Errorf("%s trim %d: offset/cap = %d/%d, want %d/%d", sc.Label, n, c.Offset, c.capacity, r.Offset, r.Capacity)
			}
		}
	}
}

func TestRotatingKVCacheBookkeepingParity(t *testing.T) {
	fx := loadKVFixture(t)
	for _, sc := range fx.Rot {
		c := NewRotatingKVCache(sc.MaxSize, sc.Keep)
		ri := 0
		for _, s := range sc.Steps {
			plan := c.Update(s)
			r := sc.Records[ri]
			ri++
			if plan.Offset != r.Offset || plan.Capacity != r.Capacity {
				t.Errorf("%s step S=%d: offset/cap = %d/%d, want %d/%d", sc.Label, s, plan.Offset, plan.Capacity, r.Offset, r.Capacity)
			}
			if r.Idx != nil && c.Idx != *r.Idx {
				t.Errorf("%s step S=%d: idx = %d, want %d", sc.Label, s, c.Idx, *r.Idx)
			}
			if r.Size != nil && c.Size() != *r.Size {
				t.Errorf("%s step S=%d: size = %d, want %d", sc.Label, s, c.Size(), *r.Size)
			}
			if r.Fetch != nil && plan.FetchEnd != *r.Fetch {
				t.Errorf("%s step S=%d: fetch = %d, want %d", sc.Label, s, plan.FetchEnd, *r.Fetch)
			}
			if r.Trimmble != nil && c.IsTrimmable() != *r.Trimmble {
				t.Errorf("%s step S=%d: trimmable = %v, want %v", sc.Label, s, c.IsTrimmable(), *r.Trimmble)
			}
		}
		for _, n := range sc.Trims {
			got := c.Trim(n)
			r := sc.Records[ri]
			ri++
			if r.Trimmed != nil && got != *r.Trimmed {
				t.Errorf("%s trim %d: trimmed = %d, want %d", sc.Label, n, got, *r.Trimmed)
			}
			if c.Offset != r.Offset {
				t.Errorf("%s trim %d: offset = %d, want %d", sc.Label, n, c.Offset, r.Offset)
			}
			if r.Idx != nil && c.Idx != *r.Idx {
				t.Errorf("%s trim %d: idx = %d, want %d", sc.Label, n, c.Idx, *r.Idx)
			}
		}
	}
}

func BenchmarkKVCacheUpdate(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		c := &KVCache{}
		c.Update(512)
		for range 256 {
			c.Update(1)
		}
	}
}

func BenchmarkRotatingKVCacheUpdate(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		c := NewRotatingKVCache(4096, 4)
		c.Update(512)
		for range 256 {
			c.Update(1)
		}
	}
}
