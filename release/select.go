// SPDX-License-Identifier: MIT OR Apache-2.0

package release

import "strings"

// Release is one entry from a GitHub /releases response, holding only the fields
// the stable-release picker reads.
type Release struct {
	TagName    string `json:"tag_name"`
	Draft      bool   `json:"draft"`
	Prerelease bool   `json:"prerelease"`
}

// SelectLatestStableRelease picks the highest stable release from a GitHub
// /releases response. The GitHub prerelease flag is not trusted on its own:
// dev/rc tags have historically shipped with it unset, so PEP 440 is applied too
// and any pre-release version is skipped, along with drafts, missing tags, and
// unparseable tags. It returns nil when nothing qualifies.
func SelectLatestStableRelease(releases []Release) *Release {
	var best *Release
	var bestVersion Version

	for i := range releases {
		r := &releases[i]
		if r.Draft || r.Prerelease {
			continue
		}
		if r.TagName == "" {
			continue
		}
		version, ok := ParseVersion(strings.TrimLeft(r.TagName, "v"))
		if !ok {
			continue
		}
		if version.IsPrerelease() {
			continue
		}
		if best == nil || version.Compare(bestVersion) > 0 {
			bestVersion = version
			best = r
		}
	}
	return best
}
