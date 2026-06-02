// SPDX-License-Identifier: MIT OR Apache-2.0

package tokenizer

import "testing"

const benchText = "The quick brown fox jumps over the lazy dog. " +
	"Pack my box with five dozen liquor jugs. " +
	"Sphinx of black quartz, judge my vow."

func BenchmarkMockEncode(b *testing.B) {
	tok := NewMock()
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.Encode(benchText)
	}
}

func BenchmarkMockDecode(b *testing.B) {
	tok := NewMock()
	ids := tok.Encode(benchText)
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.Decode(ids)
	}
}

// BenchmarkMockIncrementalDetokenize models the streaming hot path: one AddToken
// per generated token, exactly as the scheduler drives it during decode.
func BenchmarkMockIncrementalDetokenize(b *testing.B) {
	tok := NewMock()
	ids := tok.Encode(benchText)
	b.ReportAllocs()
	for b.Loop() {
		d := tok.NewIncrementalDetokenizer()
		for _, id := range ids {
			_ = d.AddToken(id)
		}
	}
}

func TestMockIncrementalMatchesDecode(t *testing.T) {
	tok := NewMock()
	ids := tok.Encode(benchText)
	d := tok.NewIncrementalDetokenizer()
	var streamed string
	for _, id := range ids {
		streamed += d.AddToken(id)
	}
	if streamed != tok.Decode(ids) {
		t.Errorf("incremental %q != Decode %q", streamed, tok.Decode(ids))
	}
	if d.Text() != streamed {
		t.Errorf("Text() %q != streamed %q", d.Text(), streamed)
	}
}
