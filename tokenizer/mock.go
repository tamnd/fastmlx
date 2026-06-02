// SPDX-License-Identifier: Apache-2.0

package tokenizer

import (
	"strings"
	"unicode/utf8"
)

// mockEOS is the mock end-of-sequence token. It sits one past the maximum valid
// Unicode code point so it can never collide with a real rune token.
const mockEOS = utf8.MaxRune + 1

// Mock is a rune-based tokenizer: every rune is its own token whose ID is the
// rune's code point. It carries no model and needs no files, so the serving
// path, scheduler, and SSE encoder run end-to-end before the cgo tokenizer
// lands. Output produced by the mock decode backend round-trips through Decode
// into readable text.
type Mock struct{}

// NewMock returns a ready-to-use rune tokenizer.
func NewMock() *Mock { return &Mock{} }

// Encode maps each rune of text to a token ID equal to its code point.
func (m *Mock) Encode(text string) []int {
	ids := make([]int, 0, len(text))
	for _, r := range text {
		ids = append(ids, int(r))
	}
	return ids
}

// Decode reassembles runes from token IDs, skipping the EOS token and any ID
// that is not a valid code point.
func (m *Mock) Decode(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		if r, ok := runeFor(id); ok {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// EOSTokenID returns the mock EOS token.
func (m *Mock) EOSTokenID() int { return mockEOS }

// VocabSize reports the rune space plus the EOS token.
func (m *Mock) VocabSize() int { return mockEOS + 1 }

// NewIncrementalDetokenizer returns a streaming detokenizer over the rune space.
func (m *Mock) NewIncrementalDetokenizer() IncrementalDetokenizer {
	return &mockDetokenizer{}
}

func runeFor(id int) (rune, bool) {
	if id < 0 || id > utf8.MaxRune {
		return 0, false
	}
	r := rune(id)
	if !utf8.ValidRune(r) {
		return 0, false
	}
	return r, true
}

type mockDetokenizer struct {
	b strings.Builder
}

func (d *mockDetokenizer) AddToken(id int) string {
	r, ok := runeFor(id)
	if !ok {
		return ""
	}
	d.b.WriteRune(r)
	return string(r)
}

func (d *mockDetokenizer) Text() string { return d.b.String() }
