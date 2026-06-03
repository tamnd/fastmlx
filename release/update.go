// SPDX-License-Identifier: MIT OR Apache-2.0

package release

import "strings"

// BuildUpdateResult shapes the admin update-check payload from a stable release
// already chosen by SelectLatestStableRelease and the running version. The
// GitHub fetch and the 24h cache stay caller seams; this is the pure decision.
//
// With no qualifying release the "no update" shape is returned. Otherwise the
// latest version is the release tag with any leading "v" stripped, and an update
// is offered only when that PEP 440 version is strictly greater than the current
// one; an unparseable version on either side is treated as no update rather than
// surfacing a bad offer. release_url is null when the release carries no URL,
// mirroring data.get("html_url") on a release object without that field.
func BuildUpdateResult(selected *Release, currentVersion string) map[string]any {
	noUpdate := map[string]any{
		"update_available": false,
		"latest_version":   nil,
		"release_url":      nil,
	}
	if selected == nil {
		return noUpdate
	}
	latest := strings.TrimLeft(selected.TagName, "v")
	latestVer, ok1 := ParseVersion(latest)
	currentVer, ok2 := ParseVersion(currentVersion)
	updateAvailable := ok1 && ok2 && latestVer.Compare(currentVer) > 0
	if !updateAvailable {
		return noUpdate
	}
	var releaseURL any
	if selected.HTMLURL != "" {
		releaseURL = selected.HTMLURL
	}
	return map[string]any{
		"update_available": true,
		"latest_version":   latest,
		"release_url":      releaseURL,
	}
}
