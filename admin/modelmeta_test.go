// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"testing"
)

type paramsCase struct {
	Params map[string]int `json:"params"`
	Out    int            `json:"out"`
}

type sizeCase struct {
	Bytes int    `json:"bytes"`
	Out   string `json:"out"`
}

type countCase struct {
	Params int    `json:"params"`
	Out    string `json:"out"`
}

type modelMetaFixture struct {
	Disk   []paramsCase `json:"disk"`
	Params []paramsCase `json:"params"`
	Sizes  []sizeCase   `json:"sizes"`
	Counts []countCase  `json:"counts"`
}

func loadModelMeta(t *testing.T) modelMetaFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/modelmeta.json")
	if err != nil {
		t.Fatal(err)
	}
	var f modelMetaFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestCalcSafetensorsDiskSizeParity(t *testing.T) {
	for i, c := range loadModelMeta(t).Disk {
		if got := CalcSafetensorsDiskSize(c.Params); got != c.Out {
			t.Errorf("CalcSafetensorsDiskSize case %d (%v) = %d, want %d", i, c.Params, got, c.Out)
		}
	}
}

func TestGetParamCountParity(t *testing.T) {
	for i, c := range loadModelMeta(t).Params {
		if got := GetParamCount(c.Params); got != c.Out {
			t.Errorf("GetParamCount case %d (%v) = %d, want %d", i, c.Params, got, c.Out)
		}
	}
}

func TestFormatModelSizeParity(t *testing.T) {
	for i, c := range loadModelMeta(t).Sizes {
		if got := FormatModelSize(c.Bytes); got != c.Out {
			t.Errorf("FormatModelSize case %d (%d) = %q, want %q", i, c.Bytes, got, c.Out)
		}
	}
}

func TestFormatParamCountParity(t *testing.T) {
	for i, c := range loadModelMeta(t).Counts {
		if got := FormatParamCount(c.Params); got != c.Out {
			t.Errorf("FormatParamCount case %d (%d) = %q, want %q", i, c.Params, got, c.Out)
		}
	}
}

func BenchmarkFormatModelSize(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = FormatModelSize(7516192768)
	}
}
