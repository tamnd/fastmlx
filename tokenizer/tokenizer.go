// SPDX-License-Identifier: MIT OR Apache-2.0

// Package tokenizer defines the tokenizer seam used by the engine and the
// scheduler. The real tokenizer is a cgo HuggingFace binding that lands with the
// compute backend (spec 1990, 06_discovery_tokenizer_settings.md). Until then the
// rune-based Mock here lets the full serving path encode prompts, detokenize
// incrementally, and produce readable text behind the mock decode backend.
package tokenizer

// Tokenizer converts between text and token IDs. The contract matches the subset
// of the HuggingFace tokenizer surface the scheduler relies on.
type Tokenizer interface {
	// Encode turns text into token IDs (no special tokens added).
	Encode(text string) []int
	// Decode turns token IDs back into text.
	Decode(ids []int) string
	// EOSTokenID is the end-of-sequence token; -1 if the tokenizer has none.
	EOSTokenID() int
	// VocabSize reports the number of distinct tokens.
	VocabSize() int
	// NewIncrementalDetokenizer returns a detokenizer that decodes a growing
	// token stream one step at a time without re-decoding the whole prefix.
	NewIncrementalDetokenizer() IncrementalDetokenizer
}

// IncrementalDetokenizer accumulates tokens and yields the new text produced by
// each step. It mirrors mlx-lm's streaming detokenizer: appending a token may
// emit zero bytes (a partial multibyte rune) and a later token flushes them.
type IncrementalDetokenizer interface {
	// AddToken appends one token and returns the text newly decodable because of
	// it. The empty string means the token did not complete any rune yet.
	AddToken(id int) string
	// Text returns everything decoded so far.
	Text() string
}
