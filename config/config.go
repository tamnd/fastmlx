// SPDX-License-Identifier: Apache-2.0

// Package config holds the resolved, immutable configuration the fastmlx server
// reads at runtime. It defines the config dataclasses (ServerConfig,
// ModelConfig, GenerationConfig, SchedulerConfig, PagedSSDCacheConfig) so an
// existing ~/.fastmlx install is a drop-in source of truth.
package config

import (
	"fmt"
	"strconv"
	"strings"
)

// ParseSize parses a human-readable size string ("100GB", "50MB", "1TB", "0")
// into bytes, including the bare-number
// (bytes) fallback. The empty string and "0" both yield 0.
func ParseSize(s string) (int64, error) {
	s = strings.ToUpper(strings.TrimSpace(s))
	if s == "" {
		return 0, nil
	}
	// Order matters: check the two-letter units before the bare "B".
	units := []struct {
		suffix string
		mult   int64
	}{
		{"TB", 1 << 40},
		{"GB", 1 << 30},
		{"MB", 1 << 20},
		{"KB", 1 << 10},
		{"B", 1},
	}
	for _, u := range units {
		if num, ok := strings.CutSuffix(s, u.suffix); ok {
			v, err := strconv.ParseFloat(num, 64)
			if err == nil {
				return int64(v * float64(u.mult)), nil
			}
		}
	}
	if v, err := strconv.ParseInt(s, 10, 64); err == nil {
		return v, nil
	}
	return 0, fmt.Errorf("invalid size string: %q", s)
}

// ServerConfig holds the server configuration.
type ServerConfig struct {
	Host        string   `json:"host"`
	Port        int      `json:"port"`
	LogLevel    string   `json:"log_level"`
	CORSOrigins []string `json:"cors_origins"`
}

// GenerationConfig holds the generation defaults.
type GenerationConfig struct {
	MaxTokens     int     `json:"max_tokens"`
	Temperature   float64 `json:"temperature"`
	TopP          float64 `json:"top_p"`
	TopK          int     `json:"top_k"`
	ForceSampling bool    `json:"force_sampling"`
}

// SchedulerConfig holds the scheduler defaults.
type SchedulerConfig struct {
	MaxNumSeqs          int   `json:"max_num_seqs"`
	CompletionBatchSize int   `json:"completion_batch_size"`
	EmbeddingBatchSize  int   `json:"embedding_batch_size"`
	StreamInterval      int   `json:"stream_interval"`
	EnableThinking      *bool `json:"enable_thinking"`
}

// PagedSSDCacheConfig configures the paged SSD cache. The server only
// supports paged SSD-based caching; sizes are human-readable strings.
type PagedSSDCacheConfig struct {
	Enabled         bool   `json:"enabled"`
	HotCacheOnly    bool   `json:"hot_cache_only"`
	CacheDir        string `json:"cache_dir"`
	MaxSize         string `json:"max_size"`           // default "100GB"
	HotCacheMaxSize string `json:"hot_cache_max_size"` // "0" = disabled
}

// MaxSizeBytes resolves MaxSize to bytes.
func (c PagedSSDCacheConfig) MaxSizeBytes() int64 {
	n, _ := ParseSize(c.MaxSize)
	return n
}

// HotCacheMaxSizeBytes resolves HotCacheMaxSize to bytes.
func (c PagedSSDCacheConfig) HotCacheMaxSizeBytes() int64 {
	n, _ := ParseSize(c.HotCacheMaxSize)
	return n
}

// DefaultGeneration returns the GenerationConfig defaults.
func DefaultGeneration() GenerationConfig {
	return GenerationConfig{MaxTokens: 32768, Temperature: 1.0, TopP: 0.95, TopK: 0}
}

// DefaultScheduler returns the SchedulerConfig defaults.
func DefaultScheduler() SchedulerConfig {
	return SchedulerConfig{MaxNumSeqs: 8, CompletionBatchSize: 8, EmbeddingBatchSize: 32, StreamInterval: 1}
}

// DefaultServer returns the ServerConfig defaults.
func DefaultServer() ServerConfig {
	return ServerConfig{Host: "0.0.0.0", Port: 8000, LogLevel: "info", CORSOrigins: []string{"*"}}
}

// DefaultCache returns the PagedSSDCacheConfig defaults.
func DefaultCache() PagedSSDCacheConfig {
	return PagedSSDCacheConfig{MaxSize: "100GB", HotCacheMaxSize: "0"}
}
