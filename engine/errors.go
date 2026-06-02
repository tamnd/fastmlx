// SPDX-License-Identifier: MIT OR Apache-2.0

package engine

import "errors"

// ErrQueueFull is returned when admission control rejects a request. The HTTP
// layer maps it to 503.
var ErrQueueFull = errors.New("scheduler queue full")

// ErrNotRunning is returned when the engine is used before Start or after Stop.
var ErrNotRunning = errors.New("engine not running")
