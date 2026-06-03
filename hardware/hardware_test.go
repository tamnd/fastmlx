// SPDX-License-Identifier: MIT OR Apache-2.0

package hardware

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type hardwareFixture struct {
	DefaultMemoryBytes int64 `json:"default_memory_bytes"`
	Chip               []struct {
		Input   string `json:"input"`
		Name    string `json:"name"`
		Variant string `json:"variant"`
	} `json:"chip"`
	MemoryGB []struct {
		Bytes int64   `json:"bytes"`
		GB    float64 `json:"gb"`
	} `json:"memory_gb"`
	WorkingSet []struct {
		TotalRAM *int64 `json:"total_ram"`
		Bytes    int64  `json:"bytes"`
	} `json:"working_set"`
	OwnerHash []struct {
		UUID     string `json:"uuid"`
		ChipName string `json:"chip_name"`
		GPUCores *int   `json:"gpu_cores"`
		MemoryGB int    `json:"memory_gb"`
		Hash     string `json:"hash"`
	} `json:"owner_hash"`
}

func loadFixture(t *testing.T) hardwareFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "hardware.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx hardwareFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	return fx
}

func TestDefaultMemoryBytes(t *testing.T) {
	fx := loadFixture(t)
	if DefaultMemoryBytes != fx.DefaultMemoryBytes {
		t.Fatalf("DefaultMemoryBytes = %d, want %d", DefaultMemoryBytes, fx.DefaultMemoryBytes)
	}
}

func TestParseChipInfo(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.Chip {
		name, variant := ParseChipInfo(c.Input)
		if name != c.Name || variant != c.Variant {
			t.Errorf("ParseChipInfo(%q) = (%q, %q), want (%q, %q)", c.Input, name, variant, c.Name, c.Variant)
		}
	}
}

func TestTotalMemoryGB(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.MemoryGB {
		if got := TotalMemoryGB(c.Bytes); got != c.GB {
			t.Errorf("TotalMemoryGB(%d) = %v, want %v", c.Bytes, got, c.GB)
		}
	}
}

func TestMaxWorkingSetFromRAM(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.WorkingSet {
		if got := MaxWorkingSetFromRAM(c.TotalRAM); got != c.Bytes {
			t.Errorf("MaxWorkingSetFromRAM(%v) = %d, want %d", c.TotalRAM, got, c.Bytes)
		}
	}
}

func TestComputeOwnerHash(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.OwnerHash {
		if got := ComputeOwnerHash(c.UUID, c.ChipName, c.GPUCores, c.MemoryGB); got != c.Hash {
			t.Errorf("ComputeOwnerHash(%q,%q,%v,%d) = %q, want %q", c.UUID, c.ChipName, c.GPUCores, c.MemoryGB, got, c.Hash)
		}
	}
}

func TestDetectHardware(t *testing.T) {
	name := "Apple M4 Pro GPU"
	hw := DetectHardware("Apple M4 Pro", 38654705664, 28991029248, &name)
	if hw.ChipName != "Apple M4 Pro" {
		t.Errorf("ChipName = %q", hw.ChipName)
	}
	if hw.TotalMemoryGB != 36.0 {
		t.Errorf("TotalMemoryGB = %v", hw.TotalMemoryGB)
	}
	if hw.MaxWorkingSetBytes != 28991029248 {
		t.Errorf("MaxWorkingSetBytes = %d", hw.MaxWorkingSetBytes)
	}
	if hw.MLXDeviceName == nil || *hw.MLXDeviceName != name {
		t.Errorf("MLXDeviceName = %v", hw.MLXDeviceName)
	}
}

func BenchmarkComputeOwnerHash(b *testing.B) {
	cores := 16
	b.ReportAllocs()
	for b.Loop() {
		_ = ComputeOwnerHash("ABC-123", "M4", &cores, 36)
	}
}
