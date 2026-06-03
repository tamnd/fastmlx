// SPDX-License-Identifier: MIT OR Apache-2.0

package formatting

import (
	"encoding/json"
	"os"
	"testing"
)

type formatCase struct {
	In  int64  `json:"in"`
	Out string `json:"out"`
}

type formatFixture struct {
	FormatBytes     []formatCase `json:"format_bytes"`
	FormatSizeDisc  []formatCase `json:"format_size_disc"`
	FormatSizeAdmin []formatCase `json:"format_size_admin"`
}

func loadFixture(t *testing.T) formatFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/formatting.json")
	if err != nil {
		t.Fatal(err)
	}
	var f formatFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestFormatBytesParity(t *testing.T) {
	for _, c := range loadFixture(t).FormatBytes {
		if got := FormatBytes(c.In); got != c.Out {
			t.Errorf("FormatBytes(%d) = %q, want %q", c.In, got, c.Out)
		}
	}
}

func TestFormatSizeParity(t *testing.T) {
	for _, c := range loadFixture(t).FormatSizeDisc {
		if got := FormatSize(c.In); got != c.Out {
			t.Errorf("FormatSize(%d) = %q, want %q", c.In, got, c.Out)
		}
	}
}

func TestFormatDiskSizeParity(t *testing.T) {
	for _, c := range loadFixture(t).FormatSizeAdmin {
		if got := FormatDiskSize(c.In); got != c.Out {
			t.Errorf("FormatDiskSize(%d) = %q, want %q", c.In, got, c.Out)
		}
	}
}

func BenchmarkFormatBytes(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = FormatBytes(16 * 1024 * 1024 * 1024)
		_ = FormatSize(1536 * 1024 * 1024)
		_ = FormatDiskSize(256 * 1024 * 1024)
	}
}
