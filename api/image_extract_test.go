// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

type imgExtractFixture struct {
	Cases []struct {
		Messages     []string `json:"messages"`
		TextMessages []string `json:"text_messages"`
		ImageURLs    []string `json:"image_urls"`
	} `json:"cases"`
}

func loadImgExtract(t *testing.T) imgExtractFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/imgextract.json")
	if err != nil {
		t.Fatal(err)
	}
	var f imgExtractFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestExtractImagesFromMessagesParity(t *testing.T) {
	for i, c := range loadImgExtract(t).Cases {
		msgs := make([]jval, len(c.Messages))
		for j, raw := range c.Messages {
			v, ok := parseOrdered(raw)
			if !ok {
				t.Fatalf("case %d: bad message JSON %q", i, raw)
			}
			msgs[j] = v
		}

		gotMsgs, gotURLs := ExtractImagesFromMessages(msgs)

		if len(gotMsgs) != len(c.TextMessages) {
			t.Errorf("case %d: got %d text messages, want %d", i, len(gotMsgs), len(c.TextMessages))
			continue
		}
		for j, m := range gotMsgs {
			if got := m.dump(); got != c.TextMessages[j] {
				t.Errorf("case %d msg %d:\n got  %s\n want %s", i, j, got, c.TextMessages[j])
			}
		}

		if len(gotURLs) != len(c.ImageURLs) {
			t.Errorf("case %d: got %d urls, want %d (%v vs %v)", i, len(gotURLs), len(c.ImageURLs), gotURLs, c.ImageURLs)
			continue
		}
		for j, u := range gotURLs {
			if u != c.ImageURLs[j] {
				t.Errorf("case %d url %d: got %q want %q", i, j, u, c.ImageURLs[j])
			}
		}
	}
}

func BenchmarkExtractImagesFromMessages(b *testing.B) {
	raw := []string{
		`{"role":"system","content":"sys"}`,
		`{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"http://x/m.png"}}],"name":"alice"}`,
	}
	msgs := make([]jval, len(raw))
	for j, r := range raw {
		msgs[j], _ = parseOrdered(r)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = ExtractImagesFromMessages(msgs)
	}
}
