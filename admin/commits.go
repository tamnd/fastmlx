// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"regexp"
	"strings"
)

// This file holds the pure engine-commit resolution cores the about/update panel
// uses to pin the engine dependencies to a commit: reading a PEP 610
// direct_url.json, projecting a generated _engine_commits.json, and parsing
// git+https URLs out of a pyproject.toml. The file reads (distribution metadata,
// the JSON file, the toml file) all stay caller seams; the parsed data and file
// text are passed in.

// pyprojectGitRe matches a "pkg @ git+https://.../repo@<sha>" dependency line,
// capturing the package name and the commit SHA.
var pyprojectGitRe = regexp.MustCompile(`"(\S+)\s*@\s*git\+https://[^@"]+@([0-9a-f]{7,40})"`)

// CommitFromDirectURL extracts the commit SHA and a commit URL from a parsed
// PEP 610 direct_url.json, returning ok=false when no commit is recorded. The
// repo URL falls back to defaultURL, loses a trailing slash, and drops a ".git"
// suffix before the commit path is appended.
func CommitFromDirectURL(directURL map[string]any, defaultURL string) (map[string]any, bool) {
	vcs, _ := directURL["vcs_info"].(map[string]any)
	commit := pyStr(vcs["commit_id"])
	if commit == "" {
		return nil, false
	}
	repoURL := defaultURL
	if v, ok := directURL["url"].(string); ok {
		repoURL = v
	}
	repoURL = strings.TrimRight(repoURL, "/")
	repoURL = strings.TrimSuffix(repoURL, ".git")
	return map[string]any{"commit": commit, "url": repoURL + "/commit/" + commit}, true
}

// CommitsFromEngineData projects a generated _engine_commits.json into the
// commit map. Each dict entry carrying a "commit" is kept; the URL comes from the
// entry, or the package default, and gains a commit path when it lacks one.
// Unlike the pyproject parser, this keeps entries whose package is unknown.
func CommitsFromEngineData(data map[string]any, packages map[string]string) map[string]map[string]any {
	result := map[string]map[string]any{}
	for pkgName, entryRaw := range data {
		entry, ok := entryRaw.(map[string]any)
		if !ok {
			continue
		}
		commitRaw, has := entry["commit"]
		if !has {
			continue
		}
		commit := pyStr(commitRaw)
		repoURL := packages[pkgName]
		if v, ok := entry["url"].(string); ok {
			repoURL = v
		}
		if !strings.Contains(repoURL, "/commit/") {
			repoURL = repoURL + "/commit/" + commit
		}
		result[pkgName] = map[string]any{"commit": commitRaw, "url": repoURL}
	}
	return result
}

// ParseCommitsFromPyproject extracts commit SHAs from git+https dependency URLs
// in a pyproject.toml, keeping only packages present in the packages map and
// building a commit URL from each package's repo base.
func ParseCommitsFromPyproject(content string, packages map[string]string) map[string]map[string]any {
	commits := map[string]map[string]any{}
	for _, m := range pyprojectGitRe.FindAllStringSubmatch(content, -1) {
		pkgName := strings.ToLower(strings.TrimSpace(m[1]))
		sha := m[2]
		if repoURL, ok := packages[pkgName]; ok {
			commits[pkgName] = map[string]any{"commit": sha, "url": repoURL + "/commit/" + sha}
		}
	}
	return commits
}
