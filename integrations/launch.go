// SPDX-License-Identifier: MIT OR Apache-2.0

package integrations

import (
	"maps"
	"strconv"
	"strings"
)

// This file holds the pure pieces of the launch path: the display command a
// tool shows for copy/paste, the environment scrub that strips the bundled
// Python variables before exec, and the non-interactive halves of the model
// picker. Discovering the binary, the curses UI, reading from the terminal, and
// exec'ing the tool stay the caller's seams.

// pythonScrubKeys are the bundled-interpreter variables a launched tool must not
// inherit: a tool that spawns its own Python would crash on init_fs_encoding if
// it picked these up from the app bundle.
var pythonScrubKeys = []string{"PYTHONHOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE"}

// ScrubbedEnv returns a copy of env with the bundled-Python variables removed.
// The input is left untouched.
func ScrubbedEnv(env map[string]string) map[string]string {
	out := make(map[string]string, len(env))
	maps.Copy(out, env)
	for _, k := range pythonScrubKeys {
		delete(out, k)
	}
	return out
}

// LaunchCommand builds the command line a tool's panel shows for launching it
// against the server. prefix is the resolved fastmlx CLI invocation, which the
// caller discovers. Every tool but Claude Code takes a model flag, defaulting to
// a placeholder when no model is chosen yet.
func LaunchCommand(prefix, tool, model string) string {
	if tool == "claude" {
		return prefix + " launch claude"
	}
	chosen := model
	if chosen == "" {
		chosen = "select-a-model"
	}
	return prefix + " launch " + tool + " --model " + chosen
}

// ModelInfo is one entry in the picker list: a model id and its optional
// context window.
type ModelInfo struct {
	ID               string
	MaxContextWindow *int
}

// AutoSelectModel resolves the picks that need no prompt: no models yields the
// empty id, a single model is chosen outright. The second return reports whether
// the caller still needs to prompt (true only when more than one model exists).
func AutoSelectModel(models []ModelInfo) (string, bool) {
	if len(models) == 0 {
		return "", false
	}
	if len(models) == 1 {
		return models[0].ID, false
	}
	return "", true
}

// FormatModelOption renders one numbered line of the fallback picker, appending
// the context window in grouped digits when the model reports one.
func FormatModelOption(index int, m ModelInfo) string {
	line := "  " + strconv.Itoa(index) + ". " + m.ID
	if intTruthy(m.MaxContextWindow) {
		line += "  [" + groupThousands(*m.MaxContextWindow) + " ctx]"
	}
	return line
}

// ParseModelChoice turns the user's numbered answer into a zero-based index,
// reporting whether it names a model in a list of n. It trims surrounding
// whitespace and rejects non-numbers and out-of-range values, the way the
// fallback prompt loops until a valid pick.
func ParseModelChoice(input string, n int) (int, bool) {
	choice, err := strconv.Atoi(strings.TrimSpace(input))
	if err != nil {
		return 0, false
	}
	idx := choice - 1
	if idx >= 0 && idx < n {
		return idx, true
	}
	return 0, false
}

// groupThousands formats an int with comma thousands separators, reproducing
// Python's `f"{n:,}"`.
func groupThousands(n int) string {
	s := strconv.Itoa(n)
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	var b strings.Builder
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(c)
	}
	if neg {
		return "-" + b.String()
	}
	return b.String()
}
