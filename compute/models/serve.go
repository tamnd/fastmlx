// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"encoding/json"
	"fmt"

	"github.com/tamnd/fastmlx/compute"
)

// NewBatchDecode assembles a continuous-batching decode backend from a
// checkpoint: it reads the model_type out of the config, loads the safetensors
// weights, dispatches to the family that serves that model_type, and wraps the
// model in a BatchGenerator. The returned *compute.BatchGenerator is a
// pipeline.DecodeStrategy, so it drops into the engine's Options.Decode in place
// of pipeline.MockDecode with no scheduler changes.
//
// configJSON is the config.json body; blob is a safetensors container (one file,
// or shards merged with compute.MergeTensors); eos is the end-of-sequence token
// id from the tokenizer or generation config. Everything here runs on the host:
// the failure surfaces only when the scheduler first calls Step and the model's
// forward reaches a kernel op (ErrMLXUnavailable on the default build).
func NewBatchDecode(configJSON, blob []byte, eos int) (*compute.BatchGenerator, error) {
	var head struct {
		ModelType string `json:"model_type"`
	}
	if err := json.Unmarshal(configJSON, &head); err != nil {
		return nil, fmt.Errorf("models: read model_type: %w", err)
	}
	if head.ModelType == "" {
		return nil, fmt.Errorf("models: config has no model_type")
	}
	weights, err := compute.LoadTensors(blob)
	if err != nil {
		return nil, err
	}
	model, err := BuildModel(head.ModelType, configJSON, weights, eos)
	if err != nil {
		return nil, err
	}
	return compute.NewBatchGenerator(model)
}
