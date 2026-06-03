// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"strconv"
	"strings"
)

// This file holds the pure model-compatibility cores the model-settings modal
// uses: formatting a cache size, detecting paroquant quantization, deciding
// whether the native MTP patch applies, and the full MTP-toggle decision. The
// config.json read, the weight-index read, and the dflash optional-import probe
// all stay route seams; the parsed config and weight-key list are passed in.

// paroquantReason is the user-facing string shown when paroquant gates a toggle.
const paroquantReason = "Not supported on paroquant models yet (compatibility not verified)"

// mtpNoHeadsReason, mtpWhitelistSuffix, and mtpMissingWeightsReason are the
// user-facing reasons the MTP-toggle decision surfaces.
const (
	mtpNoHeadsReason        = "model has no MTP heads in config"
	mtpWhitelistSuffix      = " is not on the MTP whitelist (supported: qwen3_5*, qwen3_6*, deepseek_v4*)"
	mtpMissingWeightsReason = "Config declares MTP layers but the converted weights are missing mtp.* tensors. Re-convert from HF with a converter that preserves MTP weights."
)

// FormatCacheSize formats a byte count as a compact cache size with no decimals:
// GB at or above one GB, otherwise MB. The zero-decimal rounding is half-to-even,
// matching Python's f"{x:.0f}".
func FormatCacheSize(sizeBytes int) string {
	gb := float64(sizeBytes) / (1024 * 1024 * 1024)
	if gb >= 1 {
		return strconv.FormatFloat(gb, 'f', 0, 64) + "GB"
	}
	mb := float64(sizeBytes) / (1024 * 1024)
	return strconv.FormatFloat(mb, 'f', 0, 64) + "MB"
}

// IsParoquantConfig reports whether a parsed config declares paroquant
// quantization, with the user-facing reason. A nil or non-paroquant config is
// not paroquant. The config.json read is a route seam.
func IsParoquantConfig(config map[string]any) (bool, string) {
	qcfg, _ := config["quantization_config"].(map[string]any)
	method := strings.ToLower(pyStr(qcfg["quant_method"]))
	if method == "paroquant" {
		return true, paroquantReason
	}
	return false, ""
}

// IsMTPCompatible reports whether the native MTP patch can apply: the config
// must declare MTP heads, the model type must be set, and it must be on the
// qwen3_5 / qwen3_6 / deepseek_v4 whitelist.
func IsMTPCompatible(config map[string]any, modelType string) bool {
	if !ConfigDeclaresMTP(config) {
		return false
	}
	if modelType == "" {
		return false
	}
	return strings.HasPrefix(modelType, "qwen3_5") ||
		strings.HasPrefix(modelType, "qwen3_6") ||
		strings.HasPrefix(modelType, "deepseek_v4")
}

// ModelHasMTPWeightTensors reports whether any weight-map key names an MTP
// tensor, by the "mtp." substring the route check uses. The weight-index read
// (or per-shard key enumeration) is a route seam; the keys are passed in.
func ModelHasMTPWeightTensors(weightKeys []string) bool {
	for _, k := range weightKeys {
		if strings.Contains(k, "mtp.") {
			return true
		}
	}
	return false
}

// MTPCompatForModel makes the native-MTP-toggle decision for a model whose
// config.json has been parsed into config and whose checkpoint MTP-weight
// presence is given by hasMTPWeights. The paroquant gate is applied first, then
// the config heads, whitelist, and weight checks, each with its own reason. The
// model_path / config.json existence / read-error cases are route seams.
func MTPCompatForModel(config map[string]any, hasMTPWeights bool) (bool, string) {
	if isParo, reason := IsParoquantConfig(config); isParo {
		return false, reason
	}
	rawType := config["model_type"]
	if !ConfigDeclaresMTP(config) {
		return false, mtpNoHeadsReason
	}
	if !IsMTPCompatible(config, pyStr(rawType)) {
		return false, "model_type=" + pyRepr(rawType) + mtpWhitelistSuffix
	}
	if !hasMTPWeights {
		return false, mtpMissingWeightsReason
	}
	return true, ""
}

// pyRepr renders a value the way Python's repr does for the cases the MTP reason
// needs: None for a missing value and a single-quoted form for a string.
func pyRepr(v any) string {
	switch s := v.(type) {
	case nil:
		return "None"
	case string:
		return "'" + s + "'"
	default:
		return pyStr(v)
	}
}
