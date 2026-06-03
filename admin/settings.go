// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import "strings"

// This file ports the side-effect-free validation and normalization the global
// settings update performs before it touches any runtime state: checking the SSE
// keepalive mode against its allowed set and resolving the requested model
// directories down to the effective list and primary directory. Persistence and
// the runtime apply stay the caller's seam.

// ValidateSSEKeepaliveMode reports a 400-detail string for an SSE keepalive mode
// outside the allowed set, or the empty string when the mode is valid. The modes
// are listed in the message in sorted order, matching the reference's
// sorted(valid_modes).
func ValidateSSEKeepaliveMode(mode string) string {
	switch mode {
	case "chunk", "comment", "off":
		return ""
	default:
		return "Invalid sse_keepalive_mode: " + mode + " (must be one of ['chunk', 'comment', 'off'])"
	}
}

// ResolvedModelDirs is the outcome of resolving a settings update's directory
// fields. HasUpdate is false when neither field was supplied, leaving the stored
// directories untouched. When true, NewDirs is the effective list (possibly
// empty) and Primary is its first entry, or nil when the list is empty.
type ResolvedModelDirs struct {
	HasUpdate bool
	NewDirs   []string
	Primary   *string
}

// ResolveModelDirs computes the effective model directories from a settings
// update. A supplied model_dirs list wins and is filtered to the entries that
// are non-blank once stripped (the original, unstripped value is kept); a lone
// model_dir becomes a single-entry list verbatim. With neither field present the
// result carries no update. Both fields are optional, so they arrive as pointers
// standing in for the reference's None.
func ResolveModelDirs(modelDirs *[]string, modelDir *string) ResolvedModelDirs {
	var newDirs []string
	switch {
	case modelDirs != nil:
		newDirs = []string{}
		for _, d := range *modelDirs {
			if strings.TrimSpace(d) != "" {
				newDirs = append(newDirs, d)
			}
		}
	case modelDir != nil:
		newDirs = []string{*modelDir}
	default:
		return ResolvedModelDirs{HasUpdate: false}
	}

	res := ResolvedModelDirs{HasUpdate: true, NewDirs: newDirs}
	if len(newDirs) > 0 {
		primary := newDirs[0]
		res.Primary = &primary
	}
	return res
}
