// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"crypto/subtle"
	"unicode"
	"unicode/utf8"
)

// This file holds the brand-free key-verification cores of the admin auth layer:
// constant-time API key checks and the API key format validator. Session-token
// signing, the request/cookie plumbing, and the login-redirect dependency stay
// seams in the server layer.

// VerifyAPIKey reports whether a client-supplied key matches the server key,
// using a constant-time comparison so a mismatch leaks no timing signal. An
// empty key on either side never matches.
func VerifyAPIKey(apiKey, serverAPIKey string) bool {
	if apiKey == "" || serverAPIKey == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(apiKey), []byte(serverAPIKey)) == 1
}

// VerifyAnyAPIKey reports whether a client-supplied key matches the main key or
// any sub key, each compared in constant time. The main key is checked first.
// An empty client key never matches, and empty configured keys are skipped.
func VerifyAnyAPIKey(apiKey, mainKey string, subKeys []string) bool {
	if apiKey == "" {
		return false
	}
	if mainKey != "" && subtle.ConstantTimeCompare([]byte(apiKey), []byte(mainKey)) == 1 {
		return true
	}
	for _, sk := range subKeys {
		if sk != "" && subtle.ConstantTimeCompare([]byte(apiKey), []byte(sk)) == 1 {
			return true
		}
	}
	return false
}

// ValidateAPIKey checks an API key's format: at least four characters, no
// whitespace, and printable characters only. It returns the first failure's
// message, or an empty message when the key is valid. The length is counted in
// characters (runes), and whitespace and printability follow the same Unicode
// rules CPython's str methods use.
func ValidateAPIKey(apiKey string) (bool, string) {
	if utf8.RuneCountInString(apiKey) < 4 {
		return false, "API key must be at least 4 characters"
	}
	for _, r := range apiKey {
		if pyIsSpace(r) {
			return false, "API key must not contain whitespace"
		}
	}
	for _, r := range apiKey {
		if !unicode.IsPrint(r) {
			return false, "API key must contain only printable characters"
		}
	}
	return true, ""
}

// pyWhitespace is the exact set of code points CPython's str.isspace() treats as
// whitespace: the ASCII whitespace and file/group/record/unit separators, the
// Unicode separators, and the next-line and no-break space. Go's unicode.IsSpace
// omits 0x1c-0x1f, so the validator uses this set to match str.isspace exactly.
var pyWhitespace = map[rune]struct{}{
	0x09: {}, 0x0a: {}, 0x0b: {}, 0x0c: {}, 0x0d: {},
	0x1c: {}, 0x1d: {}, 0x1e: {}, 0x1f: {}, 0x20: {},
	0x85: {}, 0xa0: {}, 0x1680: {},
	0x2000: {}, 0x2001: {}, 0x2002: {}, 0x2003: {}, 0x2004: {},
	0x2005: {}, 0x2006: {}, 0x2007: {}, 0x2008: {}, 0x2009: {}, 0x200a: {},
	0x2028: {}, 0x2029: {}, 0x202f: {}, 0x205f: {}, 0x3000: {},
}

func pyIsSpace(r rune) bool {
	_, ok := pyWhitespace[r]
	return ok
}
