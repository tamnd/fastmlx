// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"fmt"
	"sort"

	"github.com/tamnd/fastmlx/compute"
	"github.com/tamnd/fastmlx/mlxgo"
)

// ModelBuilder constructs a BatchGenerator-ready model from a config body and a
// loaded weight map. It parses the family's Args, sanitizes and wires the
// weights into the concrete model, and wraps the result in an Adapter sized by
// the config's layer count, with eos as the finish token. The weights are the
// raw loaded tensors; each builder applies its own Sanitize, matching the
// per-family LoadX path.
type ModelBuilder func(configJSON []byte, weights map[string]*mlxgo.Array, eos int) (compute.Model, error)

// modelBuilders maps a checkpoint's config.json model_type to its builder. The
// keys are the exact strings the reference dispatches on (the module it imports
// for that model_type). Every supported family's numeric Forward has landed and
// is registered here; adding a family is a single entry, with no change to
// BuildModel or the engine.
var modelBuilders = map[string]ModelBuilder{
	"qwen3_next": func(cfg []byte, w map[string]*mlxgo.Array, eos int) (compute.Model, error) {
		args, err := ParseQwen3NextArgs(cfg)
		if err != nil {
			return nil, err
		}
		m, err := NewQwen3NextModel(args, args.Sanitize(w))
		if err != nil {
			return nil, err
		}
		return NewAdapter(args.NumLayers(), eos, m.Forward, m.forwardBL), nil
	},
	"gemma4_text": func(cfg []byte, w map[string]*mlxgo.Array, eos int) (compute.Model, error) {
		args, err := ParseGemma4TextArgs(cfg)
		if err != nil {
			return nil, err
		}
		m, err := NewGemma4TextModel(args, args.Sanitize(w))
		if err != nil {
			return nil, err
		}
		return NewAdapter(args.NumLayers(), eos, m.Forward, m.forwardBL), nil
	},
	"deepseek_v3": func(cfg []byte, w map[string]*mlxgo.Array, eos int) (compute.Model, error) {
		args, err := ParseDeepseekV3Args(cfg)
		if err != nil {
			return nil, err
		}
		m, err := NewDeepseekV3Model(args, args.Sanitize(w))
		if err != nil {
			return nil, err
		}
		return NewAdapter(args.NumLayers(), eos, m.Forward, m.forwardBL), nil
	},
	"qwen3": func(cfg []byte, w map[string]*mlxgo.Array, eos int) (compute.Model, error) {
		args, err := ParseQwen3Args(cfg)
		if err != nil {
			return nil, err
		}
		m, err := NewQwen3Model(args, args.Sanitize(w))
		if err != nil {
			return nil, err
		}
		return NewAdapter(args.NumHiddenLayers, eos, m.Forward, m.forwardBL), nil
	},
	"llama": func(cfg []byte, w map[string]*mlxgo.Array, eos int) (compute.Model, error) {
		args, err := ParseLlamaArgs(cfg)
		if err != nil {
			return nil, err
		}
		m, err := NewLlamaModel(args, args.Sanitize(w))
		if err != nil {
			return nil, err
		}
		return NewAdapter(args.NumHiddenLayers, eos, m.Forward, m.forwardBL), nil
	},
	"glm4": func(cfg []byte, w map[string]*mlxgo.Array, eos int) (compute.Model, error) {
		args, err := ParseGlm4Args(cfg)
		if err != nil {
			return nil, err
		}
		m, err := NewGlm4Model(args, args.Sanitize(w))
		if err != nil {
			return nil, err
		}
		return NewAdapter(args.NumHiddenLayers, eos, m.Forward, m.forwardBL), nil
	},
	"phi3": func(cfg []byte, w map[string]*mlxgo.Array, eos int) (compute.Model, error) {
		args, err := ParsePhi4Args(cfg)
		if err != nil {
			return nil, err
		}
		m, err := NewPhi4Model(args, args.Sanitize(w))
		if err != nil {
			return nil, err
		}
		return NewAdapter(args.NumHiddenLayers, eos, m.Forward, m.forwardBL), nil
	},
	"ministral3": func(cfg []byte, w map[string]*mlxgo.Array, eos int) (compute.Model, error) {
		args, err := ParseMinistralArgs(cfg)
		if err != nil {
			return nil, err
		}
		m, err := NewMinistral3Model(args, args.Sanitize(w))
		if err != nil {
			return nil, err
		}
		return NewAdapter(args.NumHiddenLayers, eos, m.Forward, m.forwardBL), nil
	},
}

// BuildModel dispatches on a checkpoint's model_type and returns the
// BatchGenerator-ready model. eos is the end-of-sequence token id (from the
// tokenizer or generation config) that finishes a sequence. An unregistered
// model_type is a clear error naming the type, so an unsupported checkpoint fails
// at load time rather than mid-generation.
func BuildModel(modelType string, configJSON []byte, weights map[string]*mlxgo.Array, eos int) (compute.Model, error) {
	build, ok := modelBuilders[modelType]
	if !ok {
		return nil, fmt.Errorf("models: unsupported model_type %q", modelType)
	}
	return build(configJSON, weights, eos)
}

// RegisteredModelTypes returns the model_type strings BuildModel can serve, in
// sorted order. The engine uses it to report what it can load and to validate a
// checkpoint before reading its weights.
func RegisteredModelTypes() []string {
	types := make([]string, 0, len(modelBuilders))
	for t := range modelBuilders {
		types = append(types, t)
	}
	sort.Strings(types)
	return types
}
