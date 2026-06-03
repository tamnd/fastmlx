// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"strconv"
)

// This file holds the pure i18n cores: the cache-busting static-asset URL
// builder, the translate-or-key lookup, and the locale loader's parse-with-
// fallback core. The on-disk checks behind them (whether the asset exists and
// its modification time, and reading the locale files) stay caller seams.

// StaticVersionURL builds the URL for a static asset, appending the file
// modification time as a cache-busting query when the asset exists. The mtime
// is truncated toward zero, matching the int() applied to the float stat time.
func StaticVersionURL(path string, mtime float64, isFile bool) string {
	if isFile {
		return "/admin/static/" + path + "?v=" + strconv.Itoa(int(mtime))
	}
	return "/admin/static/" + path
}

// TranslateKey looks a key up in a locale dict, falling back to the key itself
// when it is absent.
func TranslateKey(locale map[string]any, key string) any {
	if v, ok := locale[key]; ok {
		return v
	}
	return key
}

// LoadLocaleText parses a locale dict from its JSON text, falling back to a
// second JSON text on failure and to an empty dict when both fail. The two file
// reads that supply the texts are caller seams.
func LoadLocaleText(primaryJSON, fallbackJSON string) map[string]any {
	var m map[string]any
	if err := json.Unmarshal([]byte(primaryJSON), &m); err == nil && m != nil {
		return m
	}
	m = nil
	if err := json.Unmarshal([]byte(fallbackJSON), &m); err == nil && m != nil {
		return m
	}
	return map[string]any{}
}
