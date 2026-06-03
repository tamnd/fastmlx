// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

type respStoreFixture struct {
	Normalize []struct {
		In  string `json:"in"`
		Out string `json:"out"`
	} `json:"normalize"`
	ConvertStored []struct {
		In  string `json:"in"`
		Out string `json:"out"`
	} `json:"convert_stored"`
	BuildRecord []struct {
		Public string `json:"public"`
		Input  string `json:"input"`
		Output string `json:"output"`
		Out    string `json:"out"`
	} `json:"build_record"`
}

func loadRespStoreFixture(t *testing.T) respStoreFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/responses_store.json")
	if err != nil {
		t.Fatal(err)
	}
	var f respStoreFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func mustParse(t *testing.T, s string) jval {
	t.Helper()
	v, ok := parseOrdered(s)
	if !ok {
		t.Fatalf("parseOrdered(%q) failed", s)
	}
	return v
}

func dumpMessages(msgs []jval) string {
	return jval{kind: kindArray, arr: msgs}.dump()
}

func TestNormalizeResponseOutputToMessagesParity(t *testing.T) {
	fx := loadRespStoreFixture(t)
	for i, c := range fx.Normalize {
		got := dumpMessages(NormalizeResponseOutputToMessages(mustParse(t, c.In)))
		if got != c.Out {
			t.Errorf("normalize[%d]:\n got  %s\n want %s", i, got, c.Out)
		}
	}
}

func TestConvertStoredResponseToMessagesParity(t *testing.T) {
	fx := loadRespStoreFixture(t)
	for i, c := range fx.ConvertStored {
		got := dumpMessages(ConvertStoredResponseToMessages(mustParse(t, c.In)))
		if got != c.Out {
			t.Errorf("convert_stored[%d]:\n got  %s\n want %s", i, got, c.Out)
		}
	}
}

func TestBuildResponseStoreRecordParity(t *testing.T) {
	fx := loadRespStoreFixture(t)
	for i, c := range fx.BuildRecord {
		public := mustParse(t, c.Public)
		input := mustParse(t, c.Input)
		output := mustParse(t, c.Output)
		got := BuildResponseStoreRecord(public, input.arr, output.arr).dump()
		if got != c.Out {
			t.Errorf("build_record[%d]:\n got  %s\n want %s", i, got, c.Out)
		}
	}
}

func BenchmarkNormalizeResponseOutputToMessages(b *testing.B) {
	out, _ := parseOrdered(`[{"type":"reasoning","summary":[{"text":"r"}]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]},{"type":"function_call","call_id":"c1","name":"f","arguments":"{\"a\":1}"}]`)
	b.ReportAllocs()
	for b.Loop() {
		_ = NormalizeResponseOutputToMessages(out)
	}
}
