// SPDX-License-Identifier: Apache-2.0

package pipeline

import (
	"testing"

	"github.com/tamnd/fastmlx/tokenizer"
)

func TestMockDecodeFinishStop(t *testing.T) {
	tok := tokenizer.NewMock()
	d := NewMockDecode(tok, "abc")
	uid, err := d.Insert(DecodeRequest{MaxTokens: 100})
	if err != nil {
		t.Fatal(err)
	}
	var got []int
	var reason string
	for d.HasActive() {
		res, err := d.Step()
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range res {
			got = append(got, r.Token)
			if r.FinishReason != "" {
				reason = r.FinishReason
				d.Remove(r.UID)
			}
		}
	}
	if reason != "stop" {
		t.Errorf("finish reason = %q, want stop", reason)
	}
	if tok.Decode(got) != "abc" {
		t.Errorf("decoded = %q, want abc", tok.Decode(got))
	}
	_ = uid
}

func TestMockDecodeFinishLength(t *testing.T) {
	tok := tokenizer.NewMock()
	d := NewMockDecode(tok, "abcdef")
	if _, err := d.Insert(DecodeRequest{MaxTokens: 3}); err != nil {
		t.Fatal(err)
	}
	var got []int
	var reason string
	for d.HasActive() {
		res, _ := d.Step()
		for _, r := range res {
			got = append(got, r.Token)
			if r.FinishReason != "" {
				reason = r.FinishReason
				d.Remove(r.UID)
			}
		}
	}
	if reason != "length" {
		t.Errorf("finish reason = %q, want length", reason)
	}
	if len(got) != 3 {
		t.Errorf("emitted %d tokens, want 3", len(got))
	}
}

func TestMockDecodeConcurrentSequences(t *testing.T) {
	tok := tokenizer.NewMock()
	d := NewMockDecode(tok, "xy")
	a, _ := d.Insert(DecodeRequest{MaxTokens: 10})
	b, _ := d.Insert(DecodeRequest{MaxTokens: 10})
	res, _ := d.Step()
	if len(res) != 2 {
		t.Fatalf("step yielded %d results, want 2", len(res))
	}
	// Both sequences advance in the same step (continuous batching).
	seen := map[int]bool{}
	for _, r := range res {
		seen[r.UID] = true
	}
	if !seen[a] || !seen[b] {
		t.Errorf("expected both uids %d,%d in step, got %v", a, b, seen)
	}
}
