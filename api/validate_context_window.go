// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"errors"
	"strconv"
)

// ValidateContextWindow guards a request whose prompt would overflow the model's
// context window, porting validate_context_window from server.py. The reference
// resolves the limit through get_max_context_window, which walks server state
// (per-model settings, then the engine pool's discovered config length, then the
// global sampling default); that resolution is server-state-bound, so the
// resolved value is injected here as maxCtx and only the pure decision is ported.
//
// maxCtx is the resolved limit, or nil when no tier produced one. The check fires
// only when the limit is truthy in the Python sense (non-nil and non-zero,
// matching `if max_ctx`) and the prompt is strictly longer than it, so a prompt
// exactly at the limit is allowed. On overflow it returns an error whose message
// reproduces the reference HTTPException detail byte for byte; otherwise it
// returns nil.
func ValidateContextWindow(numPromptTokens int, maxCtx *int) error {
	if maxCtx == nil || *maxCtx == 0 {
		return nil
	}
	if numPromptTokens > *maxCtx {
		return errors.New("Prompt too long: " + strconv.Itoa(numPromptTokens) +
			" tokens exceeds max context window of " + strconv.Itoa(*maxCtx) + " tokens")
	}
	return nil
}
