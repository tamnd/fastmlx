// SPDX-License-Identifier: MIT OR Apache-2.0

// Package formatting renders byte counts as human-readable strings. There are
// three distinct formatters in the reference, used in different places and with
// genuinely different output shapes, so all three are ported faithfully rather
// than collapsed into one.
package formatting

import (
	"fmt"
	"math"
)

// FormatBytes renders a byte count with a space before the unit and two decimal
// places for KB and above, e.g. "1.50 GB", "256.00 MB". Values below 1 KB are
// shown as a bare integer with " B". It tops out at GB: a value of several TB is
// still printed in gigabytes. This matches the helper used for memory and model
// sizing.
func FormatBytes(v int64) string {
	switch {
	case v >= 1024*1024*1024:
		return fmt.Sprintf("%.2f GB", float64(v)/(1024*1024*1024))
	case v >= 1024*1024:
		return fmt.Sprintf("%.2f MB", float64(v)/(1024*1024))
	case v >= 1024:
		return fmt.Sprintf("%.2f KB", float64(v)/1024)
	default:
		return fmt.Sprintf("%d B", v)
	}
}

// FormatSize renders a byte count with no space before the unit and two decimal
// places at every scale, including bytes, e.g. "1.00GB", "512.00B". It walks the
// unit ladder by repeated division, comparing the absolute value so negatives
// keep their sign in the output ("-1.00KB"), and falls through to PB for very
// large values. This matches the helper used when listing model file sizes.
func FormatSize(v int64) string {
	size := float64(v)
	for _, unit := range []string{"B", "KB", "MB", "GB", "TB"} {
		if math.Abs(size) < 1024.0 {
			return fmt.Sprintf("%.2f%s", size, unit)
		}
		size /= 1024.0
	}
	return fmt.Sprintf("%.2fPB", size)
}

// FormatDiskSize renders a byte count with a space before the unit and a mixed
// precision: a bare integer with " B" below 1 KB, one decimal place for KB and
// MB, and two decimal places for GB and TB (everything 1 GB and up prints in TB
// once past the TB threshold). This matches the helper used for disk and cache
// info in the admin surface.
func FormatDiskSize(v int64) string {
	switch {
	case v < 1024:
		return fmt.Sprintf("%d B", v)
	case v < 1024*1024:
		return fmt.Sprintf("%.1f KB", float64(v)/1024)
	case v < 1024*1024*1024:
		return fmt.Sprintf("%.1f MB", float64(v)/(1024*1024))
	case v < 1024*1024*1024*1024:
		return fmt.Sprintf("%.2f GB", float64(v)/(1024*1024*1024))
	default:
		return fmt.Sprintf("%.2f TB", float64(v)/(1024*1024*1024*1024))
	}
}
