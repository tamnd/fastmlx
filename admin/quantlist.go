// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"github.com/tamnd/fastmlx/discovery"
	"github.com/tamnd/fastmlx/quant"
)

// This file holds the pure projections behind the quantizable-model listing the
// oQ panel scans: deciding whether a config declares MTP layers, and building the
// info dict (and its source-model extension) the panel renders for one model. The
// directory walk, the config read, the weight-size sum, and the checkpoint
// MTP-weight probe all stay manager seams; the probe's result is passed in.

// ConfigDeclaresMTP reports whether a config declares multi-token-prediction
// layers, checking both layer-count keys at the top level and inside a nested
// text_config. A null, zero, or non-numeric value reads as no declaration.
func ConfigDeclaresMTP(config map[string]any) bool {
	tc, _ := config["text_config"].(map[string]any)
	return mtpLayerCount(config, "mtp_num_hidden_layers") > 0 ||
		mtpLayerCount(config, "num_nextn_predict_layers") > 0 ||
		mtpLayerCount(tc, "mtp_num_hidden_layers") > 0 ||
		mtpLayerCount(tc, "num_nextn_predict_layers") > 0
}

// QuantizableModelInfo builds the info dict the panel renders for one model. The
// vision-model flag reuses discovery's vision-subconfig predicate, and the
// has-MTP flag is supplied by the caller, which has combined ConfigDeclaresMTP
// with the checkpoint weight probe.
func QuantizableModelInfo(config map[string]any, name, path string, size int, hasMTP bool) map[string]any {
	tc, _ := config["text_config"].(map[string]any)

	modelType := pyStr(config["model_type"])
	if modelType == "" {
		modelType = pyStr(tc["model_type"])
	}

	_, isQuantized := config["quantization"]

	return map[string]any{
		"name":           name,
		"path":           path,
		"size":           size,
		"size_formatted": FormatSize(size),
		"model_type":     modelType,
		"is_quantized":   isQuantized,
		"is_vlm":         discovery.HasVisionSubconfig(config),
		"has_mtp_heads":  hasMTP,
	}
}

// SourceModelInfo extends QuantizableModelInfo with the fields the panel shows
// only for quantizable source models: the layer and expert counts and the
// streaming memory estimate.
func SourceModelInfo(config map[string]any, name, path string, size int, hasMTP bool) map[string]any {
	tc, _ := config["text_config"].(map[string]any)
	info := QuantizableModelInfo(config, name, path, size, hasMTP)
	info["num_layers"] = pyOr(getOr(config, "num_hidden_layers", 0), getOr(tc, "num_hidden_layers", 0))
	info["num_experts"] = getOr(config, "num_local_experts", 0)
	info["memory_streaming"] = quant.EstimateMemory(size)
	return info
}

// mtpLayerCount mirrors int(config.get(key, 0) or 0): a missing key, null, zero,
// or non-numeric value yields zero.
func mtpLayerCount(m map[string]any, key string) int {
	if m == nil {
		return 0
	}
	v := m[key]
	if !pyTruthy(v) {
		return 0
	}
	if n, ok := numToInt(v); ok {
		return n
	}
	return 0
}

// getOr returns m[key] when present, else the default, mirroring dict.get.
func getOr(m map[string]any, key string, def any) any {
	if m == nil {
		return def
	}
	if v, ok := m[key]; ok {
		return v
	}
	return def
}
