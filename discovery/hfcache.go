// SPDX-License-Identifier: MIT OR Apache-2.0

package discovery

import "strings"

// ParseHFCacheModelName extracts the model name from a Hugging Face Hub cache
// directory name of the form "models--Org--Name", returning the third "--"
// separated segment. The second return value is false when the name is not a
// cache entry: it must start with "models--" and contain at least two "--"
// separators. A well-formed entry whose name segment is empty (for example
// "models----" or "models--a--") returns an empty string with true. Any text
// after the name segment, including further "--" runs, stays in the result.
func ParseHFCacheModelName(name string) (string, bool) {
	if !strings.HasPrefix(name, "models--") || strings.Count(name, "--") < 2 {
		return "", false
	}
	parts := strings.SplitN(name, "--", 3)
	return parts[2], true
}
