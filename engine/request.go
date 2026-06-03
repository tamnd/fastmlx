// SPDX-License-Identifier: MIT OR Apache-2.0

package engine

// GetFinishReason returns the OpenAI finish-reason string for a finished
// status, or "" for a status that is not a finished state. StatusFinishedError
// is a fastmlx addition beyond the reference statuses and maps to "error" so the
// engine's unrecoverable-error path carries a reason; the other mappings match
// the reference get_finish_reason exactly.
func (s RequestStatus) GetFinishReason() string {
	switch s {
	case StatusFinishedStopped:
		return "stop"
	case StatusFinishedLength:
		return "length"
	case StatusFinishedAborted:
		return "abort"
	case StatusFinishedError:
		return "error"
	default:
		return ""
	}
}

// GetFinishReason returns the request's explicit finish reason when one is set,
// falling back to the reason implied by its status.
func (r *Request) GetFinishReason() string {
	if r.FinishReason != "" {
		return r.FinishReason
	}
	return r.Status.GetFinishReason()
}

// SetFinished marks the request finished with the given status. When reason is
// empty the status-derived reason is used, mirroring the reference set_finished.
func (r *Request) SetFinished(status RequestStatus, reason string) {
	r.Status = status
	if reason == "" {
		reason = status.GetFinishReason()
	}
	r.FinishReason = reason
}

// Less reports whether r should be scheduled before other: lower priority value
// wins, and equal priorities break the tie on arrival time (earlier first). It
// matches the reference Request.__lt__ used for priority-queue ordering.
func (r *Request) Less(other *Request) bool {
	if r.Priority != other.Priority {
		return r.Priority < other.Priority
	}
	return r.Arrival.Before(other.Arrival)
}

// Usage returns the OpenAI-compatible token usage for this output.
func (o *RequestOutput) Usage() map[string]int {
	return map[string]int{
		"prompt_tokens":     o.PromptTokens,
		"completion_tokens": o.CompletionTokens,
		"total_tokens":      o.PromptTokens + o.CompletionTokens,
	}
}
