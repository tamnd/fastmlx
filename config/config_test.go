// SPDX-License-Identifier: MIT OR Apache-2.0

package config

import "testing"

func TestParseSize(t *testing.T) {
	cases := []struct {
		in   string
		want int64
		err  bool
	}{
		{"100GB", 100 << 30, false},
		{"50MB", 50 << 20, false},
		{"1TB", 1 << 40, false},
		{"8GB", 8 << 30, false},
		{"0", 0, false},
		{"", 0, false},
		{"1024", 1024, false},
		{"1.5GB", int64(1.5 * float64(1<<30)), false},
		{"  2gb ", 2 << 30, false},
		{"notasize", 0, true},
	}
	for _, c := range cases {
		got, err := ParseSize(c.in)
		if c.err {
			if err == nil {
				t.Errorf("ParseSize(%q): expected error, got %d", c.in, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("ParseSize(%q): unexpected error %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("ParseSize(%q) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestDefaults(t *testing.T) {
	if g := DefaultGeneration(); g.MaxTokens != 32768 || g.Temperature != 1.0 || g.TopP != 0.95 {
		t.Errorf("generation defaults drifted: %+v", g)
	}
	if s := DefaultScheduler(); s.MaxNumSeqs != 8 || s.EmbeddingBatchSize != 32 {
		t.Errorf("scheduler defaults drifted: %+v", s)
	}
	c := DefaultCache()
	if b := c.MaxSizeBytes(); b != 100<<30 {
		t.Errorf("cache max size = %d, want %d", b, int64(100)<<30)
	}
	if b := c.HotCacheMaxSizeBytes(); b != 0 {
		t.Errorf("hot cache default = %d, want 0", b)
	}
}
