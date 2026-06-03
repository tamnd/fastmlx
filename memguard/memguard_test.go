// SPDX-License-Identifier: MIT OR Apache-2.0

package memguard

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type memguardFixture struct {
	SmallReserve       int64              `json:"small_reserve"`
	SmallThreshold     int64              `json:"small_threshold"`
	StaticReserveLarge map[string]int64   `json:"static_reserve_large"`
	ActiveReclaimRatio map[string]float64 `json:"active_reclaim_ratio"`
	SoftThreshold      float64            `json:"soft_threshold"`
	HardThreshold      float64            `json:"hard_threshold"`

	Normalize []struct {
		Tier string `json:"tier"`
		Norm string `json:"norm"`
	} `json:"normalize"`
	Static []struct {
		System  int64  `json:"system"`
		Tier    string `json:"tier"`
		Ceiling int64  `json:"ceiling"`
	} `json:"static"`
	Fallback []struct {
		Phys      int64 `json:"phys"`
		Available int64 `json:"available"`
		Ceiling   int64 `json:"ceiling"`
	} `json:"fallback"`
	Watermark []struct {
		Ceiling int64 `json:"ceiling"`
		Soft    int64 `json:"soft"`
		Hard    int64 `json:"hard"`
	} `json:"watermark"`
	Classify []struct {
		Current int64  `json:"current"`
		Ceiling int64  `json:"ceiling"`
		Level   string `json:"level"`
	} `json:"classify"`
	HardLimit []struct {
		Enabled         bool  `json:"enabled"`
		Static          int64 `json:"static"`
		DynamicOrCustom int64 `json:"dynamic_or_custom"`
		MetalCap        int64 `json:"metal_cap"`
		Limit           int64 `json:"limit"`
	} `json:"hard_limit"`
}

// reclaimRow decodes the reclaim cases, which carry distinct memory fields.
type reclaimRow struct {
	Phys     int64  `json:"phys"`
	Free     int64  `json:"free"`
	Inactive int64  `json:"inactive"`
	Active   int64  `json:"active"`
	Tier     string `json:"tier"`
	Ceiling  int64  `json:"ceiling"`
}

func loadFixture(t *testing.T) (memguardFixture, []reclaimRow) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "memguard.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx memguardFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	var rec struct {
		Reclaim []reclaimRow `json:"reclaim"`
	}
	if err := json.Unmarshal(raw, &rec); err != nil {
		t.Fatalf("unmarshal reclaim: %v", err)
	}
	return fx, rec.Reclaim
}

func TestConstants(t *testing.T) {
	fx, _ := loadFixture(t)
	if SmallSystemReserve != fx.SmallReserve || SmallSystemThreshold != fx.SmallThreshold {
		t.Fatalf("reserve/threshold mismatch")
	}
	if DefaultSoftThreshold != fx.SoftThreshold || DefaultHardThreshold != fx.HardThreshold {
		t.Fatalf("threshold mismatch")
	}
	for k, v := range fx.StaticReserveLarge {
		if StaticReserveLarge[k] != v {
			t.Fatalf("StaticReserveLarge[%s] = %d, want %d", k, StaticReserveLarge[k], v)
		}
	}
	for k, v := range fx.ActiveReclaimRatio {
		if ActiveReclaimRatio[k] != v {
			t.Fatalf("ActiveReclaimRatio[%s] = %v, want %v", k, ActiveReclaimRatio[k], v)
		}
	}
}

func TestNormalizeTier(t *testing.T) {
	fx, _ := loadFixture(t)
	for _, c := range fx.Normalize {
		if got := NormalizeTier(c.Tier); got != c.Norm {
			t.Errorf("NormalizeTier(%q) = %q, want %q", c.Tier, got, c.Norm)
		}
	}
}

func TestStaticCeiling(t *testing.T) {
	fx, _ := loadFixture(t)
	for _, c := range fx.Static {
		if got := StaticCeiling(c.System, c.Tier); got != c.Ceiling {
			t.Errorf("StaticCeiling(%d,%q) = %d, want %d", c.System, c.Tier, got, c.Ceiling)
		}
	}
}

func TestReclaimableCeiling(t *testing.T) {
	_, reclaim := loadFixture(t)
	for _, c := range reclaim {
		if got := ReclaimableCeiling(c.Phys, c.Free, c.Inactive, c.Active, c.Tier); got != c.Ceiling {
			t.Errorf("ReclaimableCeiling(%d,%d,%d,%d,%q) = %d, want %d", c.Phys, c.Free, c.Inactive, c.Active, c.Tier, got, c.Ceiling)
		}
	}
}

func TestDynamicCeilingFallback(t *testing.T) {
	fx, _ := loadFixture(t)
	for _, c := range fx.Fallback {
		if got := DynamicCeilingFallback(c.Phys, c.Available); got != c.Ceiling {
			t.Errorf("DynamicCeilingFallback(%d,%d) = %d, want %d", c.Phys, c.Available, got, c.Ceiling)
		}
	}
}

func TestWatermarks(t *testing.T) {
	fx, _ := loadFixture(t)
	for _, c := range fx.Watermark {
		if got := SoftBytes(c.Ceiling, DefaultSoftThreshold); got != c.Soft {
			t.Errorf("SoftBytes(%d) = %d, want %d", c.Ceiling, got, c.Soft)
		}
		if got := HardBytes(c.Ceiling, DefaultHardThreshold); got != c.Hard {
			t.Errorf("HardBytes(%d) = %d, want %d", c.Ceiling, got, c.Hard)
		}
	}
}

func TestClassifyPressure(t *testing.T) {
	fx, _ := loadFixture(t)
	for _, c := range fx.Classify {
		if got := ClassifyPressure(c.Current, c.Ceiling, DefaultSoftThreshold, DefaultHardThreshold); got != c.Level {
			t.Errorf("ClassifyPressure(%d,%d) = %q, want %q", c.Current, c.Ceiling, got, c.Level)
		}
	}
}

func TestHardLimit(t *testing.T) {
	fx, _ := loadFixture(t)
	for _, c := range fx.HardLimit {
		if got := HardLimit(c.Enabled, c.Static, c.DynamicOrCustom, c.MetalCap); got != c.Limit {
			t.Errorf("HardLimit(%v,%d,%d,%d) = %d, want %d", c.Enabled, c.Static, c.DynamicOrCustom, c.MetalCap, got, c.Limit)
		}
	}
}

func BenchmarkClassifyPressure(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = ClassifyPressure(95_000_000_000, 100_000_000_000, DefaultSoftThreshold, DefaultHardThreshold)
	}
}
