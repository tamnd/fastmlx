// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import "strings"

// This file holds the pure cores behind the log-view endpoint: the requested
// line count clamp, the predicate that picks log files out of a directory
// listing, the path-traversal guard on a requested file name, and the
// tail-over-content core (the same deque-of-last-N tail the file reader runs,
// fed text so it stays pure). The directory walk, the modification-time sort,
// and the file read itself remain caller seams.

// ClampLogLines bounds a requested line count to the inclusive range 1..10000.
func ClampLogLines(n int) int {
	return min(max(1, n), 10000)
}

// IsLogFileName reports whether a directory entry name is a server log file:
// it must start with "server" and either end in a ".log" suffix or carry a
// ".log." segment (the rotated "server.log.YYYY-MM-DD" shape).
func IsLogFileName(name string) bool {
	return strings.HasPrefix(name, "server") &&
		(pySuffix(name) == ".log" || strings.Contains(name, ".log."))
}

// IsUnsafeLogFile reports whether a requested log file name carries a path
// separator or a parent reference, which the endpoint rejects to block
// traversal outside the log directory.
func IsUnsafeLogFile(file string) bool {
	return strings.Contains(file, "/") ||
		strings.Contains(file, "\\") ||
		strings.Contains(file, "..")
}

// TailContent returns the last numLines lines of content joined back together
// along with the total line count, mirroring the file tail: text is split on
// universal newlines (\r\n and \r both translate to \n), each split line keeps
// its translated terminator except a final unterminated line, and only the
// last numLines lines are kept.
func TailContent(content string, numLines int) (string, int) {
	lines := splitUniversalLines(content)
	total := len(lines)
	start := max(total-numLines, 0)
	return strings.Join(lines[start:], ""), total
}

// pySuffix reproduces pathlib's PurePath.suffix: the substring from the final
// dot, but only when that dot is neither the first nor the last character.
func pySuffix(name string) string {
	i := strings.LastIndexByte(name, '.')
	if i > 0 && i < len(name)-1 {
		return name[i:]
	}
	return ""
}

// splitUniversalLines splits content the way Python text-mode iteration does
// under universal newlines: \n, \r, and \r\n each end a line and are emitted
// as a translated \n, and a trailing run with no terminator becomes a final
// line with none. \r and \n are single bytes that never appear inside a
// multibyte UTF-8 sequence, so byte iteration is safe.
func splitUniversalLines(content string) []string {
	lines := []string{}
	var b strings.Builder
	for i := 0; i < len(content); {
		c := content[i]
		switch c {
		case '\n':
			b.WriteByte('\n')
			lines = append(lines, b.String())
			b.Reset()
			i++
		case '\r':
			b.WriteByte('\n')
			lines = append(lines, b.String())
			b.Reset()
			if i+1 < len(content) && content[i+1] == '\n' {
				i += 2
			} else {
				i++
			}
		default:
			b.WriteByte(c)
			i++
		}
	}
	if b.Len() > 0 {
		lines = append(lines, b.String())
	}
	return lines
}
