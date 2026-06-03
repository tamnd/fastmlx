// SPDX-License-Identifier: MIT OR Apache-2.0

package discovery

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// contextLengthKeys lists the config keys checked for context length, in order.
var contextLengthKeys = []string{
	"max_position_embeddings", "max_seq_len", "max_seq_length", "seq_length", "n_positions",
}

// tokenizerMaxLengthSentinel is the tokenizer max-length sentinel (1e18):
// Transformers seeds model_max_length with int(1e30) when uncapped; anything
// above ~1e18 is treated as the sentinel, not a real context length.
const tokenizerMaxLengthSentinel = int64(1_000_000_000_000_000_000)

// modelConfig is the subset of config.json we read for classification. The
// vision-subconfig and context-length decisions read Raw directly so they can
// honour Python's key-presence and isinstance(int) semantics exactly.
type modelConfig struct {
	ModelType     string         `json:"model_type"`
	Architectures []string       `json:"architectures"`
	Raw           map[string]any `json:"-"`
}

// DiscoveredModel is a model found on disk, classified for engine routing.
type DiscoveredModel struct {
	ID            string
	Path          string
	Type          ModelType
	Engine        EngineType
	ModelTypeRaw  string
	Architectures []string
	ContextLength int // 0 = unknown
}

// NormalizeModelType lowercases and replaces '-' with '_', following the
// normalization before set membership checks.
func NormalizeModelType(s string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(s)), "-", "_")
}

// readConfig loads config.json from a model directory.
func readConfig(modelPath string) (*modelConfig, error) {
	b, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, err
	}
	raw, err := decodeNumberMap(b)
	if err != nil {
		return nil, err
	}
	var c modelConfig
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, err
	}
	c.Raw = raw
	return &c, nil
}

// decodeNumberMap unmarshals a JSON object into a map, keeping numbers as
// json.Number so integer literals stay distinguishable from floats. Python's
// isinstance(value, int) rejects a float like 32768.0; the default float64
// decode cannot, so context-length resolution relies on this.
func decodeNumberMap(b []byte) (map[string]any, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var m map[string]any
	if err := dec.Decode(&m); err != nil {
		return nil, err
	}
	return m, nil
}

// DetectModelType classifies a model directory, following the same
// precedence: architecture sets first, then normalized model_type, then the
// vision-subconfig heuristic. The directory-name embedding/reranker heuristics
// and audio detection are layered on top in later milestones; this covers the
// LLM/VLM/embedding/reranker core used by stage v0.1.
func DetectModelType(modelPath string) (ModelType, error) {
	c, err := readConfig(modelPath)
	if err != nil {
		return TypeLLM, err
	}
	dirName := strings.ToLower(filepath.Base(modelPath))

	// 1. Architecture-string classification (highest precedence).
	for _, arch := range c.Architectures {
		switch {
		case has(VLMArchitectures, arch):
			return TypeVLM, nil
		case has(EmbeddingArchitectures, arch):
			return TypeEmbedding, nil
		case has(RerankerArchitectures, arch):
			return TypeReranker, nil
		case has(MultimodalRerankerArchitectures, arch):
			if strings.Contains(dirName, "rerank") {
				return TypeReranker, nil
			}
			return TypeEmbedding, nil
		case has(CausalLMRerankerArchitectures, arch) && strings.Contains(dirName, "rerank"):
			return TypeReranker, nil
		case has(CausalLMEmbeddingArchitectures, arch) && strings.Contains(dirName, "embed"):
			return TypeEmbedding, nil
		}
	}

	// 2. Normalized model_type.
	mt := NormalizeModelType(c.ModelType)
	switch {
	case has(VLMModelTypes, mt):
		return TypeVLM, nil
	case has(EmbeddingModelTypes, mt):
		return TypeEmbedding, nil
	}

	// 3. Vision-subconfig heuristic.
	if HasVisionSubconfig(c.Raw) {
		return TypeVLM, nil
	}
	return TypeLLM, nil
}

// ReadModelContextLength resolves the model context length: try the
// top-level context keys, then nested text_config/language_config, then
// tokenizer_config.json's model_max_length (ignoring the int(1e30) sentinel).
// Returns 0 when no usable value is found.
func ReadModelContextLength(modelPath string) int {
	var config map[string]any
	if c, err := readConfig(modelPath); err == nil {
		config = c.Raw
	}
	var tc map[string]any
	if b, err := os.ReadFile(filepath.Join(modelPath, "tokenizer_config.json")); err == nil {
		tc, _ = decodeNumberMap(b)
	}
	return ContextLengthFromConfigs(config, tc)
}

func pickContextKey(m map[string]any) int {
	for _, key := range contextLengthKeys {
		if v, ok := asInt(m[key]); ok && v > 0 {
			return int(v)
		}
	}
	return 0
}

// asInt extracts an integer from a JSON-decoded value, matching Python's
// isinstance(value, int): only an integer literal qualifies, so a float like
// 32768.0 (or an exponent form) is rejected. Inputs must be decoded with
// decodeNumberMap so numbers arrive as json.Number rather than float64.
func asInt(v any) (int64, bool) {
	if n, ok := v.(json.Number); ok {
		if i, err := n.Int64(); err == nil {
			return i, true
		}
	}
	return 0, false
}
