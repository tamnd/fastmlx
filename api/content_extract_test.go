// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

type contentExtractFixture struct {
	Text []struct {
		In  json.RawMessage `json:"in"`
		Out string          `json:"out"`
	} `json:"text"`
	Multimodal []struct {
		In  json.RawMessage `json:"in"`
		Out json.RawMessage `json:"out"`
	} `json:"multimodal"`
	DropVoid []struct {
		In  json.RawMessage `json:"in"`
		Out json.RawMessage `json:"out"`
	} `json:"drop_void"`
}

func loadContentExtractFixture(t *testing.T) contentExtractFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/content_extract.json")
	if err != nil {
		t.Fatal(err)
	}
	var f contentExtractFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestExtractTextFromContentListParity(t *testing.T) {
	for i, c := range loadContentExtractFixture(t).Text {
		in := parseFixture(t, c.In)
		if got := ExtractTextFromContentList(in); got != c.Out {
			t.Errorf("text[%d]: got %q, want %q", i, got, c.Out)
		}
	}
}

func TestExtractMultimodalContentListParity(t *testing.T) {
	for i, c := range loadContentExtractFixture(t).Multimodal {
		in := parseFixture(t, c.In)
		want := parseFixture(t, c.Out).dump()
		if got := ExtractMultimodalContentList(in).dump(); got != want {
			t.Errorf("multimodal[%d]: got %s, want %s", i, got, want)
		}
	}
}

func TestDropVoidAssistantMessagesParity(t *testing.T) {
	for i, c := range loadContentExtractFixture(t).DropVoid {
		in := parseFixture(t, c.In)
		want := parseFixture(t, c.Out).dump()
		got := jval{kind: kindArray, arr: DropVoidAssistantMessages(in.arr)}.dump()
		if got != want {
			t.Errorf("drop_void[%d]: got %s, want %s", i, got, want)
		}
	}
}

func BenchmarkExtractMultimodalContentList(b *testing.B) {
	raw := `[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"u"}},{"type":"image","source":{"type":"base64","data":"XY"}}]`
	content, _ := parseOrdered(raw)
	b.ReportAllocs()
	for b.Loop() {
		_ = ExtractMultimodalContentList(content).dump()
	}
}
