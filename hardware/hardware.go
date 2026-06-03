// SPDX-License-Identifier: MIT OR Apache-2.0

// Package hardware holds the GPU-free hardware-info helpers: parsing the chip
// brand string, converting raw memory byte counts, and computing the benchmark
// identity hash. The live probes (sysctl, psutil, the Metal device query) are
// the system seam and feed these as plain inputs, the same split the netutil and
// enginepool packages use.
package hardware

import (
	"crypto/sha256"
	"encoding/hex"
	"regexp"
	"strconv"
)

// DefaultMemoryBytes is the 8 GB last-resort fallback used when no memory probe
// succeeds.
const DefaultMemoryBytes int64 = 8 * 1024 * 1024 * 1024

const ownerHashAlphabet = "0123456789abcdefghijklmnopqrstuvwxyz"

// HardwareInfo is the assembled hardware description. MLXDeviceName is nil when
// the Metal device query is unavailable.
type HardwareInfo struct {
	ChipName           string  `json:"chip_name"`
	TotalMemoryGB      float64 `json:"total_memory_gb"`
	MaxWorkingSetBytes int64   `json:"max_working_set_bytes"`
	MLXDeviceName      *string `json:"mlx_device_name"`
}

var chipRE = regexp.MustCompile(`M(\d+)\s*(Pro|Max|Ultra)?`)

// ParseChipInfo splits a sysctl brand string such as "Apple M4 Pro" into its
// chip name and variant, e.g. ("M4", "Pro"). A string with no recognizable
// M-series token falls back to ("M1", ""), matching the reference.
func ParseChipInfo(chipString string) (name, variant string) {
	m := chipRE.FindStringSubmatch(chipString)
	if m == nil {
		return "M1", ""
	}
	return "M" + m[1], m[2]
}

// TotalMemoryGB converts a raw byte count to gibibytes as a float, matching the
// reference's bytes / 1024**3.
func TotalMemoryGB(totalBytes int64) float64 {
	return float64(totalBytes) / (1024 * 1024 * 1024)
}

// MaxWorkingSetFromRAM reproduces the psutil branch of get_max_working_set_bytes:
// 75% of total RAM, truncated toward zero. A nil totalRAM (no psutil reading)
// falls back to DefaultMemoryBytes.
func MaxWorkingSetFromRAM(totalRAM *int64) int64 {
	if totalRAM == nil {
		return DefaultMemoryBytes
	}
	return int64(float64(*totalRAM) * 0.75)
}

// DetectHardware assembles a HardwareInfo from the resolved probe values: the raw
// chip brand string is used verbatim as the chip name (the reference does not run
// it through ParseChipInfo here).
func DetectHardware(chipBrand string, totalBytes, maxWorkingSet int64, deviceName *string) HardwareInfo {
	return HardwareInfo{
		ChipName:           chipBrand,
		TotalMemoryGB:      TotalMemoryGB(totalBytes),
		MaxWorkingSetBytes: maxWorkingSet,
		MLXDeviceName:      deviceName,
	}
}

// ComputeOwnerHash computes the hardware identity hash for benchmark submissions:
// SHA-256 over the concatenated fields, then a verify character appended from a
// base-36 alphabet indexed by the digit sum of the hex digest. gpuCores is nil
// when unknown, which serializes as the literal "None" to match the reference's
// f-string interpolation of a Python None.
func ComputeOwnerHash(uuid, chipName string, gpuCores *int, memoryGB int) string {
	cores := "None"
	if gpuCores != nil {
		cores = strconv.Itoa(*gpuCores)
	}
	raw := uuid + chipName + cores + strconv.Itoa(memoryGB)
	sum := sha256.Sum256([]byte(raw))
	hashHex := hex.EncodeToString(sum[:])
	verify := 0
	for i := 0; i < len(hashHex); i++ {
		verify += int(hashHex[i])
	}
	return hashHex + string(ownerHashAlphabet[verify%36])
}
