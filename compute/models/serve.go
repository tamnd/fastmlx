// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/json"
	"fmt"

	"github.com/tamnd/fastmlx/compute"
	"github.com/tamnd/fastmlx/mlxgo"
)

// NewBatchDecodeDir assembles a continuous-batching decode backend from a model
// directory: it loads the config and the (possibly sharded) safetensors weights
// with compute.LoadCheckpoint, then dispatches and wraps them exactly like
// NewBatchDecode. This is the call the engine uses to bring up a non-mock engine
// from a checkpoint on disk. eos is the end-of-sequence token id from the
// tokenizer or generation config.
func NewBatchDecodeDir(dir string, eos int) (*compute.BatchGenerator, error) {
	configJSON, weights, err := compute.LoadCheckpoint(dir)
	if err != nil {
		return nil, err
	}
	return newBatchDecode(configJSON, weights, eos)
}

// NewBatchDecode assembles a decode backend from a single in-memory checkpoint:
// the config.json body and one safetensors container (one file, or shards merged
// with compute.MergeTensors). The returned *compute.BatchGenerator is a
// pipeline.DecodeStrategy, so it drops into the engine's Options.Decode in place
// of pipeline.MockDecode with no scheduler changes.
//
// Everything here runs on the host: the failure surfaces only when the scheduler
// first calls Step and the model's forward reaches a kernel op (ErrMLXUnavailable
// on the default build).
func NewBatchDecode(configJSON, blob []byte, eos int) (*compute.BatchGenerator, error) {
	weights, err := compute.LoadTensors(blob)
	if err != nil {
		return nil, err
	}
	return newBatchDecode(configJSON, weights, eos)
}

// EOSFromConfig extracts the end-of-sequence token id from a config.json body.
// HuggingFace configs spell eos_token_id either as a single integer or as a list
// (the model stops on any of them); this returns the integer, or the first of the
// list, or -1 when the field is absent or null. The caller treats -1 as "the
// tokenizer or generation config must supply it instead."
func EOSFromConfig(configJSON []byte) int {
	var head struct {
		EOSTokenID json.RawMessage `json:"eos_token_id"`
	}
	if err := json.Unmarshal(configJSON, &head); err != nil || len(head.EOSTokenID) == 0 || string(head.EOSTokenID) == "null" {
		return -1
	}
	var one int
	if err := json.Unmarshal(head.EOSTokenID, &one); err == nil {
		return one
	}
	var many []int
	if err := json.Unmarshal(head.EOSTokenID, &many); err == nil && len(many) > 0 {
		return many[0]
	}
	return -1
}

// newBatchDecode is the shared core: read the model_type out of the config,
// dispatch to the family that serves it, and wrap the model in a BatchGenerator.
func newBatchDecode(configJSON []byte, weights map[string]*mlxgo.Array, eos int) (*compute.BatchGenerator, error) {
	var head struct {
		ModelType string `json:"model_type"`
	}
	if err := json.Unmarshal(configJSON, &head); err != nil {
		return nil, fmt.Errorf("models: read model_type: %w", err)
	}
	if head.ModelType == "" {
		return nil, fmt.Errorf("models: config has no model_type")
	}
	model, err := BuildModel(head.ModelType, configJSON, weights, eos)
	if err != nil {
		return nil, err
	}
	return compute.NewBatchGenerator(model)
}
