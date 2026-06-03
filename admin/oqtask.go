// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"strconv"
	"strings"
)

// This file holds the pure cores of the oQ-quantization task panel: formatting a
// byte count, turning a worker phase string into a human label, and projecting a
// task into the JSON shape the panel polls. The async quantization run, progress
// polling, and cancellation all stay manager seams.

// QuantTask is the subset of a quantization task's state that the panel serializes.
type QuantTask struct {
	TaskID      string
	ModelName   string
	ModelPath   string
	OQLevel     float64
	OutputName  string
	OutputPath  string
	Status      string
	Progress    float64
	Phase       string
	Error       string
	CreatedAt   float64
	StartedAt   float64
	CompletedAt float64
	SourceSize  int
	OutputSize  int
	Dtype       string
}

// ToDict projects the task into the panel's wire shape, rounding the progress
// percentage to one decimal the way the reference does.
func (t QuantTask) ToDict() map[string]any {
	return map[string]any{
		"task_id":      t.TaskID,
		"model_name":   t.ModelName,
		"model_path":   t.ModelPath,
		"oq_level":     t.OQLevel,
		"output_name":  t.OutputName,
		"output_path":  t.OutputPath,
		"status":       t.Status,
		"progress":     pyRound(t.Progress, 1),
		"phase":        t.Phase,
		"error":        t.Error,
		"created_at":   t.CreatedAt,
		"started_at":   t.StartedAt,
		"completed_at": t.CompletedAt,
		"source_size":  t.SourceSize,
		"output_size":  t.OutputSize,
		"dtype":        t.Dtype,
	}
}

// FormatSize renders a byte count as a human-readable string with a bytes tier
// below 1 KB, then KB/MB/GB by magnitude with one decimal. This is the task
// panel's variant, distinct from FormatModelSize in carrying the raw-bytes tier.
func FormatSize(sizeBytes int) string {
	switch {
	case sizeBytes < 1024:
		return strconv.Itoa(sizeBytes) + " B"
	case sizeBytes < 1024*1024:
		return formatOneDecimal(float64(sizeBytes)/1024) + " KB"
	case sizeBytes < 1024*1024*1024:
		return formatOneDecimal(float64(sizeBytes)/(1024*1024)) + " MB"
	default:
		return formatOneDecimal(float64(sizeBytes)/(1024*1024*1024)) + " GB"
	}
}

// PhaseLabel turns a worker phase string into a human-readable label. The fixed
// phases map to set strings (with the quantizing label naming the oQ level); an
// unknown phase is returned unchanged. A "quantizing_eta|current|total|eta"
// progress marker is parsed into a percentage and an optional remaining-time
// suffix, with a non-numeric or missing current/total reading as zero percent.
func PhaseLabel(phase string, oqLevel float64) string {
	const etaPrefix = "quantizing_eta|"
	if strings.HasPrefix(phase, etaPrefix) {
		parts := strings.Split(phase, "|")
		current := partOr(parts, 1, "?")
		total := partOr(parts, 2, "?")
		eta := ""
		if len(parts) > 3 {
			eta = parts[3]
		}
		pct := 0
		if isASCIIDigits(current) && isASCIIDigits(total) {
			c, _ := strconv.Atoi(current)
			tot, _ := strconv.Atoi(total)
			if tot < 1 {
				tot = 1
			}
			pct = int(float64(c) / float64(tot) * 100)
		}
		label := "oQ" + formatG(oqLevel) + ": " + strconv.Itoa(pct) + "%"
		if eta != "" {
			label += " (" + eta + " remaining)"
		}
		return label
	}

	switch phase {
	case "loading":
		return "Loading model..."
	case "quantizing":
		return "Quantizing to oQ" + formatG(oqLevel) + "..."
	case "saving":
		return "Saving quantized model..."
	default:
		return phase
	}
}

// partOr returns parts[i] when present, else the fallback, mirroring the
// guarded index access in the reference.
func partOr(parts []string, i int, fallback string) string {
	if len(parts) > i {
		return parts[i]
	}
	return fallback
}

// isASCIIDigits reports whether s is a non-empty run of ASCII digits, matching
// the controlled input str.isdigit() guards in the reference.
func isASCIIDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// formatOneDecimal renders a float with one decimal place, matching Python's
// f"{x:.1f}" (both round half-to-even).
func formatOneDecimal(x float64) string {
	return strconv.FormatFloat(x, 'f', 1, 64)
}

// formatG renders a float the way Python's f"{x:g}" does for the small, exact oQ
// level values in play, dropping a trailing ".0" and keeping real fractions.
func formatG(x float64) string {
	return strconv.FormatFloat(x, 'g', -1, 64)
}
