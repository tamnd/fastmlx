// SPDX-License-Identifier: MIT OR Apache-2.0

package compute

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/tamnd/fastmlx/mlxgo"
)

// ConfigFileName is the model config a checkpoint directory carries.
const ConfigFileName = "config.json"

// SafetensorsIndexName is the shard index a multi-file checkpoint carries.
const SafetensorsIndexName = "model.safetensors.index.json"

// LoadCheckpoint reads a model directory and returns the config.json body and
// the merged weight map ready for a model builder. It handles both on-disk
// layouts: a sharded checkpoint with a model.safetensors.index.json (each shard
// the index names is loaded once and merged, in sorted order), and an unsharded
// checkpoint (every *.safetensors file in the directory is loaded and merged).
// The weights are raw tensors; the caller's family Sanitize runs at build time.
//
// All of this is host-side file work and array bookkeeping; nothing here needs
// the GPU. A missing directory, a missing config, an empty or unreadable shard
// set, or a name duplicated across shards is reported rather than silently
// producing a partial model.
func LoadCheckpoint(dir string) (configJSON []byte, weights map[string]*mlxgo.Array, err error) {
	configJSON, err = os.ReadFile(filepath.Join(dir, ConfigFileName))
	if err != nil {
		return nil, nil, fmt.Errorf("checkpoint: read config: %w", err)
	}

	files, err := shardFiles(dir)
	if err != nil {
		return nil, nil, err
	}

	shards := make([]map[string]*mlxgo.Array, 0, len(files))
	for _, f := range files {
		blob, err := os.ReadFile(f)
		if err != nil {
			return nil, nil, fmt.Errorf("checkpoint: read shard %s: %w", filepath.Base(f), err)
		}
		w, err := LoadTensors(blob)
		if err != nil {
			return nil, nil, fmt.Errorf("checkpoint: load shard %s: %w", filepath.Base(f), err)
		}
		shards = append(shards, w)
	}

	weights, err = MergeTensors(shards...)
	if err != nil {
		return nil, nil, err
	}
	return configJSON, weights, nil
}

// shardFiles returns the absolute paths of the safetensors files to load, in a
// stable order. When an index is present it drives the set (so files unrelated
// to the model are not pulled in); otherwise every *.safetensors in the
// directory is used.
func shardFiles(dir string) ([]string, error) {
	indexPath := filepath.Join(dir, SafetensorsIndexName)
	if data, err := os.ReadFile(indexPath); err == nil {
		idx, err := ParseSafetensorsIndex(data)
		if err != nil {
			return nil, fmt.Errorf("checkpoint: %w", err)
		}
		files := make([]string, len(idx.Shards))
		for i, name := range idx.Shards {
			files[i] = filepath.Join(dir, name)
		}
		return files, nil
	}

	matches, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("checkpoint: scan safetensors: %w", err)
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("checkpoint: no safetensors files in %s", dir)
	}
	sort.Strings(matches)
	return matches, nil
}
