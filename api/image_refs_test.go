// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

type imageRefsFixture struct {
	Refs []struct {
		In           json.RawMessage `json:"in"`
		TextMessages json.RawMessage `json:"text_messages"`
		URLs         []string        `json:"urls"`
	} `json:"refs"`
}

func loadImageRefs(t *testing.T) imageRefsFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/image_refs.json")
	if err != nil {
		t.Fatal(err)
	}
	var f imageRefsFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestExtractImageRefsFromMessagesParity(t *testing.T) {
	for i, c := range loadImageRefs(t).Refs {
		in, ok := parseOrdered(string(c.In))
		if !ok {
			t.Fatalf("case %d: bad input fixture", i)
		}
		gotMsgs, gotURLs := ExtractImageRefsFromMessages(in.arr)

		want, ok := parseOrdered(string(c.TextMessages))
		if !ok {
			t.Fatalf("case %d: bad text_messages fixture", i)
		}
		got := jval{kind: kindArray, arr: gotMsgs}
		if got.dumpASCII() != want.dumpASCII() {
			t.Errorf("case %d text_messages:\n got  %s\n want %s", i, got.dumpASCII(), want.dumpASCII())
		}

		if len(gotURLs) != len(c.URLs) {
			t.Errorf("case %d urls: got %v, want %v", i, gotURLs, c.URLs)
			continue
		}
		for j := range gotURLs {
			if gotURLs[j] != c.URLs[j] {
				t.Errorf("case %d url %d: got %q, want %q", i, j, gotURLs[j], c.URLs[j])
			}
		}
	}
}

func BenchmarkExtractImageRefsFromMessages(b *testing.B) {
	in, _ := parseOrdered(`[{"role":"user","content":[` +
		`{"type":"text","text":"look"},` +
		`{"type":"image_url","image_url":{"url":"https://x/a.png"}},` +
		`{"type":"text","text":"here"}]},` +
		`{"role":"assistant","content":"reply"}]`)
	msgs := in.arr
	b.ReportAllocs()
	for b.Loop() {
		_, _ = ExtractImageRefsFromMessages(msgs)
	}
}
