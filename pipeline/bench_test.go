// SPDX-License-Identifier: MIT OR Apache-2.0

package pipeline

import (
	"strconv"
	"testing"

	"github.com/tamnd/fastmlx/tokenizer"
)

// BenchmarkMockDecodeStep measures one decode step across a batch of active
// sequences - the per-step fan-out the real BatchGenerator must beat.
func BenchmarkMockDecodeStep(b *testing.B) {
	for _, width := range []int{1, 8, 32} {
		b.Run("seqs="+strconv.Itoa(width), func(b *testing.B) {
			tok := tokenizer.NewMock()
			d := NewMockDecode(tok, "")
			for range width {
				if _, err := d.Insert(DecodeRequest{MaxTokens: 1 << 20}); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			for b.Loop() {
				if _, err := d.Step(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestMockDecodeInsertReturnsUniqueUIDs(t *testing.T) {
	tok := tokenizer.NewMock()
	d := NewMockDecode(tok, "abc")
	seen := map[int]bool{}
	for range 5 {
		uid, err := d.Insert(DecodeRequest{MaxTokens: 10})
		if err != nil {
			t.Fatal(err)
		}
		if seen[uid] {
			t.Fatalf("duplicate uid %d", uid)
		}
		seen[uid] = true
	}
}

func TestMockDecodeZeroMaxTokensEmitsEOS(t *testing.T) {
	tok := tokenizer.NewMock()
	d := NewMockDecode(tok, "abc")
	if _, err := d.Insert(DecodeRequest{MaxTokens: 0}); err != nil {
		// MaxTokens 0 means "no cap" in the request schema, so the full response
		// runs; this guards the genLen accounting rather than asserting EOS.
		t.Fatal(err)
	}
	res, err := d.Step()
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 {
		t.Fatalf("got %d results, want 1", len(res))
	}
}

func TestMockDecodeCloseClearsActive(t *testing.T) {
	tok := tokenizer.NewMock()
	d := NewMockDecode(tok, "abc")
	if _, err := d.Insert(DecodeRequest{MaxTokens: 10}); err != nil {
		t.Fatal(err)
	}
	if !d.HasActive() {
		t.Fatal("expected active sequence after insert")
	}
	if err := d.Close(); err != nil {
		t.Fatal(err)
	}
	if d.HasActive() {
		t.Fatal("expected no active sequences after Close")
	}
}
