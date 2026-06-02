// SPDX-License-Identifier: MIT OR Apache-2.0

// Package auth holds the GPU-free admin authentication surface: API-key format
// validation and constant-time key verification against a main key and its sub
// keys. The session-token layer (signed, timed cookies) is its own subsystem and
// is not ported here.
package auth

import (
	"crypto/subtle"
	"unicode"
)

// Session token lifetimes, in seconds. The remember-me flag swaps the 24-hour
// default for 30 days. They describe the eventual signed-cookie session layer.
const (
	SessionMaxAge    = 86400   // 24 hours
	RememberMeMaxAge = 2592000 // 30 days
)

// API-key format validation messages. These are returned verbatim.
const (
	errTooShort     = "API key must be at least 4 characters"
	errWhitespace   = "API key must not contain whitespace"
	errNotPrintable = "API key must contain only printable characters"
)

// SubKey is one sub API key for API-only authentication, alongside the main key.
type SubKey struct {
	Key       string `json:"key"`
	Name      string `json:"name"`
	CreatedAt string `json:"created_at"`
}

// ValidateAPIKey checks an API key's format: at least 4 characters, no
// whitespace, printable characters only. The second result is the empty string
// when valid, otherwise the reason. The checks run in this order, so a short key
// reports its length before anything else.
func ValidateAPIKey(apiKey string) (bool, string) {
	if len([]rune(apiKey)) < 4 {
		return false, errTooShort
	}
	for _, r := range apiKey {
		if isSpace(r) {
			return false, errWhitespace
		}
	}
	for _, r := range apiKey {
		if !isPrintable(r) {
			return false, errNotPrintable
		}
	}
	return true, ""
}

// VerifyAPIKey reports whether a presented key matches the server key, compared
// in constant time. An empty key on either side never matches.
func VerifyAPIKey(apiKey, serverKey string) bool {
	if apiKey == "" || serverKey == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(apiKey), []byte(serverKey)) == 1
}

// VerifyAnyAPIKey reports whether a presented key matches the main key or any
// sub key, each compared in constant time. An empty presented key never matches;
// empty configured keys are skipped.
func VerifyAnyAPIKey(apiKey, mainKey string, subKeys []SubKey) bool {
	if apiKey == "" {
		return false
	}
	if mainKey != "" && subtle.ConstantTimeCompare([]byte(apiKey), []byte(mainKey)) == 1 {
		return true
	}
	for _, sk := range subKeys {
		if sk.Key != "" && subtle.ConstantTimeCompare([]byte(apiKey), []byte(sk.Key)) == 1 {
			return true
		}
	}
	return false
}

// isSpace mirrors Python's str.isspace() for a single rune. Go's unicode.IsSpace
// already covers the ASCII whitespace, NEL, NBSP, and the Unicode White_Space
// property; Python additionally treats the C0 information separators (0x1c-0x1f)
// as whitespace, so they are added here.
func isSpace(r rune) bool {
	if r >= 0x1c && r <= 0x1f {
		return true
	}
	return unicode.IsSpace(r)
}

// isPrintable mirrors Python's str.isprintable() for a single rune: a character
// is non-printable when its category is Other (C) or Separator (Z), with the
// ASCII space special-cased as printable. (Unassigned code points, category Cn,
// are outside Go's tables and read as printable here; realistic keys never carry
// them, and whitespace is already rejected before this check runs.)
func isPrintable(r rune) bool {
	if r == ' ' {
		return true
	}
	return !unicode.In(r, unicode.C, unicode.Z)
}
