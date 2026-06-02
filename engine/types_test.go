// SPDX-License-Identifier: MIT OR Apache-2.0

package engine

import (
	"errors"
	"testing"
)

func TestRequestStatusFinished(t *testing.T) {
	unfinished := []RequestStatus{StatusWaiting, StatusPrefilling, StatusRunning, StatusPreempted}
	finished := []RequestStatus{StatusFinishedStopped, StatusFinishedLength, StatusFinishedAborted, StatusFinishedError}
	for _, s := range unfinished {
		if s.Finished() {
			t.Errorf("status %d should not be finished", s)
		}
	}
	for _, s := range finished {
		if !s.Finished() {
			t.Errorf("status %d should be finished", s)
		}
	}
}

func TestStatusOrdering(t *testing.T) {
	// The Finished() boundary relies on the finished states sorting last.
	if StatusFinishedStopped <= StatusRunning {
		t.Fatal("finished states must enumerate after active states")
	}
}

func TestSentinelsDistinct(t *testing.T) {
	if errors.Is(ErrQueueFull, ErrNotRunning) || errors.Is(ErrNotRunning, ErrQueueFull) {
		t.Fatal("ErrQueueFull and ErrNotRunning must be distinct sentinels")
	}
	wrapped := errors.Join(errors.New("ctx"), ErrQueueFull)
	if !errors.Is(wrapped, ErrQueueFull) {
		t.Fatal("ErrQueueFull must be detectable through errors.Is when wrapped")
	}
}
