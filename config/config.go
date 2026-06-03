// SPDX-License-Identifier: MIT OR Apache-2.0

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

// ModelConfig holds the model selection. ModelPath is optional; a nil pointer
// means no explicit path was set. TrustRemoteCode defaults to false: a model
// repo can ship arbitrary modeling code that runs at load time, so the flag
// must be turned on deliberately per model.
type ModelConfig struct {
	ModelName       string  `json:"model_name"`
	TrustRemoteCode bool    `json:"trust_remote_code"`
	ModelPath       *string `json:"model_path"`
}

// MCPConfig holds the Model Context Protocol settings. ConfigPath is optional;
// a nil pointer means none was set.
type MCPConfig struct {
	ConfigPath *string `json:"config_path"`
	Enabled    bool    `json:"enabled"`
}

// CacheConfig is retained for structural compatibility. All cache options live
// on PagedSSDCacheConfig; this section carries no fields.
type CacheConfig struct{}

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

// DefaultModel returns the ModelConfig defaults.
func DefaultModel() ModelConfig {
	return ModelConfig{}
}

// DefaultMCP returns the MCPConfig defaults.
func DefaultMCP() MCPConfig {
	return MCPConfig{}
}

// Config is the centralized configuration that combines every section. The
// continuous-batching flag is a top-level feature toggle.
type Config struct {
	Server             ServerConfig        `json:"server"`
	Model              ModelConfig         `json:"model"`
	Generation         GenerationConfig    `json:"generation"`
	Scheduler          SchedulerConfig     `json:"scheduler"`
	Cache              CacheConfig         `json:"cache"`
	PagedSSDCache      PagedSSDCacheConfig `json:"paged_ssd_cache"`
	MCP                MCPConfig           `json:"mcp"`
	ContinuousBatching bool                `json:"continuous_batching"`
}

// DefaultConfig returns a Config with every section at its defaults.
func DefaultConfig() Config {
	return Config{
		Server:        DefaultServer(),
		Model:         DefaultModel(),
		Generation:    DefaultGeneration(),
		Scheduler:     DefaultScheduler(),
		PagedSSDCache: DefaultCache(),
		MCP:           DefaultMCP(),
	}
}

// Validate checks the configuration and returns a list of error messages, one
// per problem found, in section order: server, generation, then paged SSD
// cache. An empty slice means the configuration is valid.
func (c Config) Validate() []string {
	errors := []string{}

	// Server: the port must be a usable TCP port number.
	if !(0 < c.Server.Port && c.Server.Port < 65536) {
		errors = append(errors, fmt.Sprintf("Invalid port: %d", c.Server.Port))
	}

	// Generation: token budget and sampling ranges.
	if c.Generation.MaxTokens <= 0 {
		errors = append(errors, fmt.Sprintf("max_tokens must be positive: %d", c.Generation.MaxTokens))
	}
	if !(0.0 <= c.Generation.Temperature && c.Generation.Temperature <= 2.0) {
		errors = append(errors, "temperature must be 0.0-2.0: "+pyFloat(c.Generation.Temperature))
	}
	if !(0.0 <= c.Generation.TopP && c.Generation.TopP <= 1.0) {
		errors = append(errors, "top_p must be 0.0-1.0: "+pyFloat(c.Generation.TopP))
	}

	// Paged SSD cache: a cache directory is required once it is enabled.
	if c.PagedSSDCache.Enabled && c.PagedSSDCache.CacheDir == "" {
		errors = append(errors, "Paged SSD cache enabled but no cache_dir specified")
	}

	return errors
}

// pyFloat formats f the way Python's str() renders a float, so validation
// messages match the reference byte for byte. The shortest round-tripping form
// is used, and an integer-valued float keeps a trailing ".0" (Python prints
// 5.0, not 5). Exponent thresholds line up with Python repr for the small
// magnitudes that appear in generation settings.
func pyFloat(f float64) string {
	s := strconv.FormatFloat(f, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eEnN") {
		s += ".0"
	}
	return s
}
