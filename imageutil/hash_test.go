// SPDX-License-Identifier: MIT OR Apache-2.0

package imageutil

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type imageHashFixture struct {
	Cases []struct {
		Images []struct {
			Width  int    `json:"width"`
			Height int    `json:"height"`
			RGBB64 string `json:"rgb_b64"`
		} `json:"images"`
		Combined *string  `json:"combined"`
		PerImage []string `json:"per_image"`
	} `json:"cases"`
}

func loadImageHash(t *testing.T) imageHashFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/imagehash.json")
	if err != nil {
		t.Fatal(err)
	}
	var f imageHashFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestImageHashParity(t *testing.T) {
	for i, c := range loadImageHash(t).Cases {
		images := make([]Image, len(c.Images))
		for j, im := range c.Images {
			rgb, err := base64.StdEncoding.DecodeString(im.RGBB64)
			if err != nil {
				t.Fatal(err)
			}
			images[j] = Image{Width: im.Width, Height: im.Height, RGB: rgb}
		}

		wantCombined := ""
		if c.Combined != nil {
			wantCombined = *c.Combined
		}
		if got := ComputeImageHash(images); got != wantCombined {
			t.Errorf("case %d combined: got %q want %q", i, got, wantCombined)
		}

		got := ComputePerImageHashes(images)
		want := c.PerImage
		if len(got) == 0 && len(want) == 0 {
			continue
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("case %d per-image:\n got  %v\n want %v", i, got, want)
		}
	}
}

func TestComputePerImageHashesEmpty(t *testing.T) {
	if got := ComputePerImageHashes(nil); len(got) != 0 {
		t.Errorf("empty input: got %v want empty", got)
	}
}

func makeRGB(w, h, seed int) []byte {
	b := make([]byte, w*h*3)
	for i := range b {
		b[i] = byte((seed*31 + i*7) % 256)
	}
	return b
}

func BenchmarkComputeImageHash(b *testing.B) {
	images := []Image{
		{Width: 64, Height: 64, RGB: makeRGB(64, 64, 1)},
		{Width: 32, Height: 48, RGB: makeRGB(32, 48, 2)},
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = ComputeImageHash(images)
	}
}

func BenchmarkComputePerImageHashes(b *testing.B) {
	images := []Image{
		{Width: 64, Height: 64, RGB: makeRGB(64, 64, 1)},
		{Width: 32, Height: 48, RGB: makeRGB(32, 48, 2)},
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = ComputePerImageHashes(images)
	}
}
