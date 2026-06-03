// SPDX-License-Identifier: MIT OR Apache-2.0

package quant

import (
	"maps"
	"strings"
)

// vlmTextPrefix is the model-tree prefix VLM text towers carry. oQ writes
// per-layer quantization overrides keyed by the safetensors tensor base name
// (e.g. "lm_head"), but the quantize class predicate is handed model-tree paths
// (e.g. "language_model.lm_head"), so without the prefixed variant the lookup
// misses and the global bits are used, which then mismatches at load_weights.
const vlmTextPrefix = "language_model."

// ExpandPerLayerQuantKeys adds the language_model.-prefixed (and de-prefixed)
// variants of per-layer quantization overrides so both the tensor-name and the
// model-tree-path lookups resolve. For each of the "quantization" and
// "quantization_config" blocks, every dict-valued entry whose key lacks the
// prefix gains a prefixed alias, and every prefixed key gains its bare alias,
// but only when that alias is not already present. Non-dict blocks and non-dict
// entries are left alone. The config is mutated in place and returned.
func ExpandPerLayerQuantKeys(cfg map[string]any) map[string]any {
	for _, configKey := range [...]string{"quantization", "quantization_config"} {
		quant, ok := cfg[configKey].(map[string]any)
		if !ok {
			continue
		}
		extras := map[string]any{}
		for key, val := range quant {
			if _, ok := val.(map[string]any); !ok {
				continue
			}
			if !strings.HasPrefix(key, vlmTextPrefix) {
				prefixed := vlmTextPrefix + key
				if _, present := quant[prefixed]; !present {
					extras[prefixed] = val
				}
			} else {
				short := key[len(vlmTextPrefix):]
				if _, present := quant[short]; !present {
					extras[short] = val
				}
			}
		}
		maps.Copy(quant, extras)
	}
	return cfg
}
