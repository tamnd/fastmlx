// SPDX-License-Identifier: MIT OR Apache-2.0

// Package memguard holds the GPU-free decision arithmetic of the process memory
// enforcer: how the per-tier reserves and reclaim ratios turn raw system memory
// readings into a hard ceiling, soft/hard watermarks, and a pressure level. The
// live readings (Metal active bytes, the macOS vm_stat host statistics, the
// iogpu wired limit, psutil) are the system seam and feed these as plain inputs,
// the same split the memory and enginepool packages use.
package memguard

import "strings"

// Byte reserves and the small-system cutoff. Systems below the threshold get a
// flat reserve regardless of tier, since a tier-scaled cut would leave too
// little to load a useful model.
const (
	SmallSystemReserve   int64 = 4 * 1024 * 1024 * 1024
	SmallSystemThreshold int64 = 16 * 1024 * 1024 * 1024
)

// Default watermark fractions of the hard ceiling.
const (
	DefaultSoftThreshold = 0.90
	DefaultHardThreshold = 0.95
)

// StaticReserveLarge is the per-tier static reserve for systems at or above the
// small-system threshold. custom shares a small reserve so the static cap stays
// sane regardless of the user-entered ceiling.
var StaticReserveLarge = map[string]int64{
	"safe":       12 * 1024 * 1024 * 1024,
	"balanced":   8 * 1024 * 1024 * 1024,
	"aggressive": 6 * 1024 * 1024 * 1024,
	"custom":     2 * 1024 * 1024 * 1024,
}

// ActiveReclaimRatio is the fraction of "active" pages counted as reclaimable
// via macOS compression/swap, per tier (custom uses a user ceiling instead).
var ActiveReclaimRatio = map[string]float64{
	"safe":       0.2,
	"balanced":   0.5,
	"aggressive": 0.8,
}

// NormalizeTier lowercases and trims a tier name, falling back to "balanced" for
// anything not in StaticReserveLarge.
func NormalizeTier(tier string) string {
	t := strings.ToLower(strings.TrimSpace(tier))
	if _, ok := StaticReserveLarge[t]; !ok {
		return "balanced"
	}
	return t
}

// StaticCeiling is total RAM minus the tier-scaled static reserve, clamped at
// zero. custom always uses its own reserve; other tiers on a small system use
// the flat SmallSystemReserve. tier is expected already normalized.
func StaticCeiling(systemBytes int64, tier string) int64 {
	if tier == "custom" {
		return clampZero(systemBytes - StaticReserveLarge["custom"])
	}
	var reserve int64
	if systemBytes < SmallSystemThreshold {
		reserve = SmallSystemReserve
	} else {
		reserve = StaticReserveLarge[tier]
	}
	return clampZero(systemBytes - reserve)
}

// ReclaimableCeiling is the safe/balanced/aggressive dynamic ceiling: the
// process footprint plus free and inactive memory plus the reclaimable fraction
// of active memory, clamped at zero. active*ratio truncates toward zero.
func ReclaimableCeiling(physFootprint, free, inactive, active int64, tier string) int64 {
	ratio := ActiveReclaimRatio[tier]
	return clampZero(physFootprint + free + inactive + int64(float64(active)*ratio))
}

// DynamicCeilingFallback is the non-macOS / vm_stat-failure dynamic ceiling: the
// process footprint plus psutil-reported available memory, clamped at zero.
func DynamicCeilingFallback(physFootprint, available int64) int64 {
	return clampZero(physFootprint + available)
}

// SoftBytes is the soft watermark, ceiling*softThreshold truncated, or 0 when the
// ceiling is non-positive.
func SoftBytes(ceiling int64, softThreshold float64) int64 {
	if ceiling <= 0 {
		return 0
	}
	return int64(float64(ceiling) * softThreshold)
}

// HardBytes is the hard watermark, ceiling*hardThreshold truncated, or 0 when the
// ceiling is non-positive.
func HardBytes(ceiling int64, hardThreshold float64) int64 {
	if ceiling <= 0 {
		return 0
	}
	return int64(float64(ceiling) * hardThreshold)
}

// ClassifyPressure maps current usage against the watermarks to "ok", "soft", or
// "hard". A non-positive ceiling (guard disabled) is always "ok".
func ClassifyPressure(current, ceiling int64, softThreshold, hardThreshold float64) string {
	if ceiling <= 0 {
		return "ok"
	}
	soft := SoftBytes(ceiling, softThreshold)
	hard := HardBytes(ceiling, hardThreshold)
	switch {
	case current < soft:
		return "ok"
	case current < hard:
		return "soft"
	default:
		return "hard"
	}
}

// HardLimit is the final ceiling: the minimum of the static ceiling, the
// resolved dynamic-or-custom ceiling, and (when positive) the Metal cap. It
// returns 0 when the guard is disabled, which callers treat as "no limit".
func HardLimit(enabled bool, staticCeiling, dynamicOrCustom, metalCap int64) int64 {
	if !enabled {
		return 0
	}
	limit := min(staticCeiling, dynamicOrCustom)
	if metalCap > 0 && metalCap < limit {
		limit = metalCap
	}
	return limit
}

func clampZero(v int64) int64 {
	if v < 0 {
		return 0
	}
	return v
}
