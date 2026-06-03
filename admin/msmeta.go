// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"strconv"
	"strings"
)

// This file holds the pure ModelScope-metadata cores the download panel uses to
// enrich a model entry: summing weight-file sizes, estimating a parameter count
// from a model config, and normalizing a raw API entry into the panel shape. The
// network fetches (file listing, config.json, model detail) are caller seams.

// ExtractModelSizeFromFiles sums the byte sizes from a file-metadata listing,
// taking each file's "Size" or "size" field (whichever is truthy) and ignoring
// entries whose size is missing or not a number.
func ExtractModelSizeFromFiles(fileList []map[string]any) int {
	total := 0
	for _, f := range fileList {
		size := f["Size"]
		if !pyTruthy(size) {
			size = f["size"]
		}
		if !pyTruthy(size) {
			continue
		}
		if n, ok := numToInt(size); ok {
			total += n
		}
	}
	return total
}

// EstimateParamsFromConfig estimates a decoder-transformer's parameter count
// from an HF-style config, covering dense Llama/Qwen/Mistral families and MoE
// variants. It returns 0 when a required field is missing or non-numeric, since
// the caller renders a blank rather than a wrong number. The estimate is a rough
// headline figure, not a byte-exact count.
func EstimateParamsFromConfig(config map[string]any) int {
	if config == nil {
		return 0
	}

	vocabSize, ok1 := cfgInt(config, "vocab_size", 0)
	hiddenSize, ok2 := cfgInt(config, "hidden_size", 0)
	numLayers, ok3 := cfgInt(config, "num_hidden_layers", 0)
	if !ok1 || !ok2 || !ok3 {
		return 0
	}
	if vocabSize == 0 || hiddenSize == 0 || numLayers == 0 {
		return 0
	}

	intermediateSize, ok := cfgInt(config, "intermediate_size", 0)
	if !ok {
		return 0
	}
	numHeads, ok := cfgInt(config, "num_attention_heads", 0)
	if !ok {
		return 0
	}
	numKV, ok := cfgInt(config, "num_key_value_heads", numHeads)
	if !ok {
		return 0
	}
	headDim, ok := cfgInt(config, "head_dim", 0)
	if !ok {
		return 0
	}
	if headDim == 0 && numHeads != 0 {
		headDim = hiddenSize / numHeads
	}

	// int(num_local_experts or num_experts or 1): the first truthy of the two
	// config fields, defaulting to a single (dense) expert.
	expertsVal := config["num_local_experts"]
	if !pyTruthy(expertsVal) {
		expertsVal = config["num_experts"]
	}
	if !pyTruthy(expertsVal) {
		expertsVal = 1
	}
	numExperts, ok := numToInt(expertsVal)
	if !ok {
		return 0
	}

	tieEmbeddings := true
	if v, present := config["tie_word_embeddings"]; present {
		tieEmbeddings = pyTruthy(v)
	}

	embeddings := vocabSize * hiddenSize

	// Attention: Q and O span the full hidden size; K and V are reduced for GQA.
	var attn int
	if numHeads != 0 && headDim != 0 {
		attn = 2*hiddenSize*(numHeads*headDim) + 2*hiddenSize*(numKV*headDim)
	} else {
		attn = 4 * hiddenSize * hiddenSize
	}

	// Gated MLP (gate + up + down), multiplied by the expert count for MoE.
	var ffn int
	if intermediateSize != 0 {
		ffn = numExperts * 3 * hiddenSize * intermediateSize
	} else {
		ffn = 8 * hiddenSize * hiddenSize
	}

	layerNorms := 2 * hiddenSize
	perLayer := attn + ffn + layerNorms

	total := embeddings + numLayers*perLayer + hiddenSize
	if !tieEmbeddings {
		total += vocabSize * hiddenSize // untied LM head
	}
	return total
}

// ParseMSModelEntry normalizes a raw ModelScope API model entry into the panel's
// model shape, building the "owner/name" repo id from the Path and Name fields,
// reading the download and like counts (falling back from Likes to Stars), and
// formatting the storage size when present. Parameter fields are left null for
// the enrichment step to fill.
func ParseMSModelEntry(entry map[string]any) map[string]any {
	path := pyStr(entry["Path"])
	name := pyStr(entry["Name"])

	var modelID string
	switch {
	case path != "" && name != "":
		modelID = path + "/" + name
	case name != "":
		modelID = name
	default:
		modelID = path
	}

	downloads := pyOr(entry["Downloads"], 0)
	likes := pyOr(pyOrAny(entry["Likes"], entry["Stars"]), 0)
	size := pyOr(entry["StorageSize"], 0)

	displayName := name
	if displayName == "" {
		parts := strings.Split(modelID, "/")
		displayName = parts[len(parts)-1]
	}

	sizeFormatted := ""
	if n, ok := numToInt(size); ok && n > 0 {
		sizeFormatted = FormatModelSize(n)
	}

	return map[string]any{
		"repo_id":          modelID,
		"name":             displayName,
		"downloads":        downloads,
		"likes":            likes,
		"trending_score":   0,
		"size":             size,
		"size_formatted":   sizeFormatted,
		"params":           nil,
		"params_formatted": nil,
	}
}

// cfgInt mirrors int(config.get(key, default)): a missing key yields the default,
// a present number or numeric string is coerced like Python's int (floats
// truncate toward zero), and a present null or non-numeric value fails so the
// caller can return its zero result.
func cfgInt(config map[string]any, key string, def int) (int, bool) {
	v, present := config[key]
	if !present {
		return def, true
	}
	return numToInt(v)
}

// numToInt coerces a JSON-decoded value the way Python's int() would, reporting
// ok=false for a null or non-numeric value (where int() raises).
func numToInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case bool:
		if n {
			return 1, true
		}
		return 0, true
	case string:
		i, err := strconv.Atoi(strings.TrimSpace(n))
		if err != nil {
			return 0, false
		}
		return i, true
	default:
		return 0, false
	}
}

// pyTruthy reports the Python truth value of a JSON-decoded value: zero numbers,
// empty strings and collections, false, and null are falsy.
func pyTruthy(v any) bool {
	switch n := v.(type) {
	case nil:
		return false
	case bool:
		return n
	case float64:
		return n != 0
	case int:
		return n != 0
	case string:
		return n != ""
	case []any:
		return len(n) > 0
	case map[string]any:
		return len(n) > 0
	default:
		return true
	}
}

// pyStr returns a string value or "", mirroring entry.get(key) or "".
func pyStr(v any) string {
	if s, ok := v.(string); ok && s != "" {
		return s
	}
	return ""
}

// pyOr returns v when truthy, else the fallback, mirroring `v or fallback`.
func pyOr(v any, fallback any) any {
	if pyTruthy(v) {
		return v
	}
	return fallback
}

// pyOrAny returns the first of two values that is truthy, or the second,
// mirroring `a or b`.
func pyOrAny(a, b any) any {
	if pyTruthy(a) {
		return a
	}
	return b
}
