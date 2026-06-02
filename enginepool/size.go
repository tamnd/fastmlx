// SPDX-License-Identifier: MIT OR Apache-2.0

// Package enginepool manages multiple model engines with LRU-based eviction
// under a memory ceiling, model pinning, and pre-load admission. The portable
// core ported here is the LRU/pinning bookkeeping, the eviction-victim
// selection, the admission loop, and the on-disk model-size estimation. The
// actual engine instantiation and the live Metal/process memory probing are
// injected seams (callbacks/interfaces), so the algorithm is deterministic and
// testable before the compute backend lands.
package enginepool

import (
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// FormatSize renders a byte count as a human-readable string, matching the
// reference format_size: two decimals, climbing through B/KB/MB/GB/TB and
// falling back to PB, dividing by 1024 at each step.
func FormatSize(sizeBytes int64) string {
	v := float64(sizeBytes)
	for _, unit := range []string{"B", "KB", "MB", "GB", "TB"} {
		if math.Abs(v) < 1024.0 {
			return fmt.Sprintf("%.2f%s", v, unit)
		}
		v /= 1024.0
	}
	return fmt.Sprintf("%.2fPB", v)
}

// EstimateModelSize estimates a model's memory footprint from its on-disk
// weight files, matching the reference estimate_model_size. MLX keeps quantized
// weights compressed, so file size approximates memory use. It sums the
// top-level .safetensors files; failing that, the top-level .bin files (skipping
// optimizer/training artifacts); failing that, .safetensors anywhere in the
// tree. A 5% runtime-buffer overhead is added. An empty total is an error.
func EstimateModelSize(modelPath string) (int64, error) {
	var total int64

	// Primary: top-level safetensors files.
	matches, _ := filepath.Glob(filepath.Join(modelPath, "*.safetensors"))
	for _, f := range matches {
		if info, err := os.Stat(f); err == nil {
			total += info.Size()
		}
	}

	// Fallback: top-level .bin files (older PyTorch format), minus non-weights.
	if total == 0 {
		bins, _ := filepath.Glob(filepath.Join(modelPath, "*.bin"))
		for _, f := range bins {
			name := strings.ToLower(filepath.Base(f))
			if strings.Contains(name, "optimizer") || strings.Contains(name, "training") {
				continue
			}
			if info, err := os.Stat(f); err == nil {
				total += info.Size()
			}
		}
	}

	// Fallback: safetensors stored in subdirectories.
	if total == 0 {
		_ = filepath.WalkDir(modelPath, func(path string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			if strings.HasSuffix(path, ".safetensors") {
				if info, e := d.Info(); e == nil {
					total += info.Size()
				}
			}
			return nil
		})
	}

	if total == 0 {
		return 0, fmt.Errorf("No model weights found in %s", modelPath)
	}

	// Add overhead for runtime buffers (~5%).
	return int64(float64(total) * 1.05), nil
}
