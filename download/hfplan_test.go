// SPDX-License-Identifier: MIT OR Apache-2.0

package download

import (
	"encoding/json"
	"os"
	"testing"
)

type hfFixture struct {
	DtypeBytes map[string]int64  `json:"dtype_bytes"`
	SortMap    map[string]string `json:"sort_map"`
	DiskSize   []struct {
		St struct {
			Parameters map[string]int64 `json:"parameters"`
		} `json:"st"`
		Bytes  int64 `json:"bytes"`
		Params int64 `json:"params"`
	} `json:"disk_size"`
	FormatSize []struct {
		Bytes int64  `json:"bytes"`
		Out   string `json:"out"`
	} `json:"format_size"`
	FormatParam []struct {
		N   int64  `json:"n"`
		Out string `json:"out"`
	} `json:"format_param"`
	RepoID []struct {
		In      string  `json:"in"`
		OK      bool    `json:"ok"`
		Cleaned *string `json:"cleaned"`
		Error   string  `json:"error"`
	} `json:"repo_id"`
}

func loadFixture(t *testing.T) hfFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/hfdownload.json")
	if err != nil {
		t.Fatal(err)
	}
	var f hfFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestDtypeBytesAndSortMapParity(t *testing.T) {
	fx := loadFixture(t)
	for k, v := range fx.DtypeBytes {
		if DtypeBytes[k] != v {
			t.Errorf("DtypeBytes[%q] = %d, want %d", k, DtypeBytes[k], v)
		}
	}
	if len(DtypeBytes) != len(fx.DtypeBytes) {
		t.Errorf("DtypeBytes size %d, want %d", len(DtypeBytes), len(fx.DtypeBytes))
	}
	for k, v := range fx.SortMap {
		got, ok := SortField(k)
		if !ok || got != v {
			t.Errorf("SortField(%q) = (%q,%v), want %q", k, got, ok, v)
		}
	}
	if len(SortMap) != len(fx.SortMap) {
		t.Errorf("SortMap size %d, want %d", len(SortMap), len(fx.SortMap))
	}
	if _, ok := SortField("nope"); ok {
		t.Error("SortField(unknown) should be !ok")
	}
}

func TestSafetensorsSizeParity(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.DiskSize {
		if got := SafetensorsDiskSize(c.St.Parameters); got != c.Bytes {
			t.Errorf("SafetensorsDiskSize(%v) = %d, want %d", c.St.Parameters, got, c.Bytes)
		}
		if got := ParamCount(c.St.Parameters); got != c.Params {
			t.Errorf("ParamCount(%v) = %d, want %d", c.St.Parameters, got, c.Params)
		}
	}
}

func TestFormatModelSizeParity(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.FormatSize {
		if got := FormatModelSize(c.Bytes); got != c.Out {
			t.Errorf("FormatModelSize(%d) = %q, want %q", c.Bytes, got, c.Out)
		}
	}
}

func TestFormatParamCountParity(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.FormatParam {
		if got := FormatParamCount(c.N); got != c.Out {
			t.Errorf("FormatParamCount(%d) = %q, want %q", c.N, got, c.Out)
		}
	}
}

func TestValidateRepoIDParity(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.RepoID {
		cleaned, err := ValidateRepoID(c.In)
		if c.OK {
			if err != nil {
				t.Errorf("ValidateRepoID(%q) errored: %v", c.In, err)
				continue
			}
			if c.Cleaned != nil && cleaned != *c.Cleaned {
				t.Errorf("ValidateRepoID(%q) cleaned = %q, want %q", c.In, cleaned, *c.Cleaned)
			}
		} else {
			if err == nil {
				t.Errorf("ValidateRepoID(%q) should have errored", c.In)
				continue
			}
			if err.Error() != c.Error {
				t.Errorf("ValidateRepoID(%q) err = %q, want %q", c.In, err.Error(), c.Error)
			}
		}
	}
}

func TestNormalizeEndpoint(t *testing.T) {
	cases := map[string]string{
		"https://huggingface.co":   "https://huggingface.co",
		"https://huggingface.co/":  "https://huggingface.co",
		"https://hf-mirror.com///": "https://hf-mirror.com",
		"":                         "",
	}
	for in, want := range cases {
		if got := NormalizeEndpoint(in); got != want {
			t.Errorf("NormalizeEndpoint(%q) = %q, want %q", in, got, want)
		}
	}
}

func BenchmarkFormatModelSize(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = FormatModelSize(13000000000)
	}
}
