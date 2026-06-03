// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"strconv"
	"strings"
)

// UploadResponse is the slice of a failed community-benchmark upload response
// the error sanitizer reads: the cf-mitigated header value, the raw body text,
// and the status code (HasStatus false stands for a response without one). The
// HTTP call that produces it is the caller's seam.
type UploadResponse struct {
	CFMitigated string
	Body        string
	StatusCode  int
	HasStatus   bool
}

// SanitizeUploadError turns a failed upload response into a short, presentable
// error string instead of letting a raw HTML body (such as a Cloudflare
// challenge page) reach the dashboard's error column. Resolution order: a
// Cloudflare challenge (authoritative cf-mitigated header, or a body sniff for
// the interstitial markers that covers transports stripping the header); then a
// JSON error envelope (the API's normal shape: error, then detail, then
// message), truncated to 300 characters; then a short plain-text body, with an
// HTML-looking body collapsed to a one-line byte-count hint; finally the bare
// status code.
func SanitizeUploadError(r UploadResponse) string {
	status := "?"
	if r.HasStatus {
		status = strconv.Itoa(r.StatusCode)
	}

	cf := strings.ToLower(r.CFMitigated)
	bodyHead := runeHead(strings.ToLower(r.Body), 512)
	if cf == "challenge" || strings.Contains(bodyHead, "just a moment") || strings.Contains(bodyHead, "cf-chl") {
		return "Upload blocked by Cloudflare (HTTP " + status + "). " +
			"This is a server-side issue with fastmlx.ai — retry later or " +
			"report it to the maintainer."
	}

	var data map[string]any
	if err := json.Unmarshal([]byte(r.Body), &data); err == nil && data != nil {
		msg := pyOrAny(pyOrAny(data["error"], data["detail"]), data["message"])
		if s, ok := msg.(string); ok && s != "" {
			return runeHead(s, 300)
		}
	}

	text := strings.TrimSpace(r.Body)
	if strings.Contains(text, "<") && strings.Contains(text, ">") {
		return "HTTP " + status + " — unexpected non-JSON response (" + strconv.Itoa(len([]rune(r.Body))) + " bytes)"
	}
	if head := runeHead(text, 300); head != "" {
		return head
	}
	return "HTTP " + status
}

// runeHead returns the first n characters of s, counting by rune to match
// Python's character-based slicing rather than Go's byte slicing.
func runeHead(s string, n int) string {
	runes := []rune(s)
	if len(runes) > n {
		return string(runes[:n])
	}
	return s
}
