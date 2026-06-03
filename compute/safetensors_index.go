// SPDX-License-Identifier: MIT OR Apache-2.0

package compute

import (
	"encoding/json"
	"fmt"
	"sort"
)

// SafetensorsIndex is the parsed model.safetensors.index.json that maps each
// weight name to the shard file that holds it. WeightMap is the raw mapping;
// Shards is the deduplicated, sorted list of shard file names; Metadata is the
// optional metadata block (e.g. total_size), kept as raw JSON values since its
// fields vary.
type SafetensorsIndex struct {
	WeightMap map[string]string
	Shards    []string
	Metadata  map[string]json.RawMessage
}

// ShardFor returns the shard file for a weight and whether it was mapped.
func (idx *SafetensorsIndex) ShardFor(weight string) (string, bool) {
	f, ok := idx.WeightMap[weight]
	return f, ok
}

// ParseSafetensorsIndex parses a sharded-model index. It requires a non-empty
// weight_map; the derived Shards list is sorted and deduplicated so the loader
// can open each file once.
func ParseSafetensorsIndex(data []byte) (*SafetensorsIndex, error) {
	var doc struct {
		Metadata  map[string]json.RawMessage `json:"metadata"`
		WeightMap map[string]string          `json:"weight_map"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("safetensors index: %w", err)
	}
	if len(doc.WeightMap) == 0 {
		return nil, fmt.Errorf("safetensors index: empty or missing weight_map")
	}
	seen := make(map[string]bool)
	var shards []string
	for _, f := range doc.WeightMap {
		if !seen[f] {
			seen[f] = true
			shards = append(shards, f)
		}
	}
	sort.Strings(shards)
	return &SafetensorsIndex{
		WeightMap: doc.WeightMap,
		Shards:    shards,
		Metadata:  doc.Metadata,
	}, nil
}
