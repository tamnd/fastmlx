// SPDX-License-Identifier: MIT OR Apache-2.0

package tokenizer

import "testing"

func TestMockRoundTrip(t *testing.T) {
	m := NewMock()
	cases := []string{"", "hello", "héllo wörld", "日本語のテスト", "emoji 🚀 ok"}
	for _, s := range cases {
		if got := m.Decode(m.Encode(s)); got != s {
			t.Errorf("round-trip %q -> %q", s, got)
		}
	}
}

func TestMockDecodeSkipsEOS(t *testing.T) {
	m := NewMock()
	ids := m.Encode("hi")
	ids = append(ids, m.EOSTokenID())
	if got := m.Decode(ids); got != "hi" {
		t.Errorf("decode with EOS = %q, want %q", got, "hi")
	}
}

func TestMockIncrementalDetokenizer(t *testing.T) {
	m := NewMock()
	d := m.NewIncrementalDetokenizer()
	ids := m.Encode("héllo")
	var streamed string
	for _, id := range ids {
		streamed += d.AddToken(id)
	}
	if streamed != "héllo" {
		t.Errorf("streamed = %q, want %q", streamed, "héllo")
	}
	if d.Text() != "héllo" {
		t.Errorf("Text() = %q, want %q", d.Text(), "héllo")
	}
	// EOS contributes no text.
	if got := d.AddToken(m.EOSTokenID()); got != "" {
		t.Errorf("AddToken(EOS) = %q, want empty", got)
	}
}
