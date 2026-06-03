// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"strings"
)

// This file holds the pure cores of the model-upload panel: deciding whether a
// folder name marks an oQ model, deciding whether a README carries real content
// beyond YAML frontmatter, and projecting an upload task into the panel's wire
// shape. The folder read, the Hub upload, and the model-card generation all stay
// caller seams.

// UploadTask is the subset of an upload task's state that the panel serializes.
type UploadTask struct {
	TaskID      string
	ModelName   string
	ModelPath   string
	RepoID      string
	Status      string
	Progress    float64
	Error       string
	CreatedAt   float64
	StartedAt   float64
	CompletedAt float64
	TotalSize   int
	RepoURL     string
}

// ToDict projects the task into the panel's wire shape, rounding the progress to
// one decimal and formatting the total size (blank when the size is zero). The
// size uses the no-bytes-tier formatter the uploader shares with the model list.
func (t UploadTask) ToDict() map[string]any {
	totalSizeFormatted := ""
	if t.TotalSize != 0 {
		totalSizeFormatted = FormatModelSize(t.TotalSize)
	}
	return map[string]any{
		"task_id":              t.TaskID,
		"model_name":           t.ModelName,
		"model_path":           t.ModelPath,
		"repo_id":              t.RepoID,
		"status":               t.Status,
		"progress":             pyRound(t.Progress, 1),
		"error":                t.Error,
		"created_at":           t.CreatedAt,
		"started_at":           t.StartedAt,
		"completed_at":         t.CompletedAt,
		"total_size":           t.TotalSize,
		"total_size_formatted": totalSizeFormatted,
		"repo_url":             t.RepoURL,
	}
}

// IsOQModel reports whether a folder name marks an oQ-quantized model, matching
// the reference's case-sensitive "oQ" substring test.
func IsOQModel(name string) bool {
	return strings.Contains(name, "oQ")
}

// HasMeaningfulReadme reports whether a README carries content beyond YAML
// frontmatter. The caller passes the file text (or "" when the file is absent or
// unreadable, both of which are not meaningful). A document that is only
// frontmatter, or has an unclosed opening marker, is not meaningful; any text
// after a closed frontmatter block, or any document without frontmatter, is.
func HasMeaningfulReadme(text string) bool {
	text = strings.TrimFunc(text, pyIsSpace)
	if text == "" {
		return false
	}
	if strings.HasPrefix(text, "---") {
		parts := strings.SplitN(text, "---", 3)
		if len(parts) >= 3 {
			body := strings.TrimFunc(parts[2], pyIsSpace)
			return body != ""
		}
		return false
	}
	return true
}
