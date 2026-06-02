// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"

	"github.com/tamnd/fastmlx/enginecore"
	"github.com/tamnd/fastmlx/scheduler"
	"github.com/tamnd/fastmlx/server"
)

// serveOptions is the `serve` command flag set.
type serveOptions struct {
	modelDir              string
	host                  string
	port                  int
	logLevel              string
	sseKeepaliveMode      string
	maxConcurrentRequests int
	embeddingBatchSize    int
	pagedSSDCacheDir      string
	pagedSSDCacheMaxSize  string
	hotCacheMaxSize       string
	noCache               bool
	initialCacheBlocks    int
	mcpConfig             string
	hfEndpoint            string
	msEndpoint            string
	httpProxy             string
	httpsProxy            string
	noProxy               string
	caBundle              string
	basePath              string
	apiKey                string
}

func runServe(args []string) error {
	home, _ := os.UserHomeDir()
	defaultBase := filepath.Join(home, ".fastmlx")

	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	o := serveOptions{}
	fs.StringVar(&o.modelDir, "model-dir", "", "model directory (default: <base>/models)")
	fs.StringVar(&o.host, "host", "127.0.0.1", "bind address")
	fs.IntVar(&o.port, "port", 8000, "port")
	fs.StringVar(&o.logLevel, "log-level", "info", "log level (trace|debug|info|warning|error)")
	fs.StringVar(&o.sseKeepaliveMode, "sse-keepalive-mode", "chunk", "SSE keepalive (chunk|comment|off)")
	fs.IntVar(&o.maxConcurrentRequests, "max-concurrent-requests", 8, "max concurrent requests")
	fs.IntVar(&o.embeddingBatchSize, "embedding-batch-size", 32, "embedding batch size")
	fs.StringVar(&o.pagedSSDCacheDir, "paged-ssd-cache-dir", "", "SSD cache directory")
	fs.StringVar(&o.pagedSSDCacheMaxSize, "paged-ssd-cache-max-size", "100GB", "SSD cache max size")
	fs.StringVar(&o.hotCacheMaxSize, "hot-cache-max-size", "0", "hot cache max size (0 = disabled)")
	fs.BoolVar(&o.noCache, "no-cache", false, "disable the paged SSD cache")
	fs.IntVar(&o.initialCacheBlocks, "initial-cache-blocks", 256, "pre-allocated cache blocks")
	fs.StringVar(&o.mcpConfig, "mcp-config", "", "MCP config file (JSON/YAML)")
	fs.StringVar(&o.hfEndpoint, "hf-endpoint", "", "custom HuggingFace endpoint")
	fs.StringVar(&o.msEndpoint, "ms-endpoint", "", "custom ModelScope endpoint")
	fs.StringVar(&o.httpProxy, "http-proxy", "", "HTTP proxy")
	fs.StringVar(&o.httpsProxy, "https-proxy", "", "HTTPS proxy")
	fs.StringVar(&o.noProxy, "no-proxy", "", "bypass proxy for hosts")
	fs.StringVar(&o.caBundle, "ca-bundle", "", "CA bundle PEM file")
	fs.StringVar(&o.basePath, "base-path", defaultBase, "base directory")
	fs.StringVar(&o.apiKey, "api-key", "", "API key for authentication")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if o.modelDir == "" {
		o.modelDir = filepath.Join(o.basePath, "models")
	}

	// v0.2: the serving layer runs behind the mock decode backend so the
	// OpenAI-compatible HTTP path is exercisable end-to-end. The compute backend
	// (real tokens) lands in v0.4 (spec 1990, 02_compute_backend_mlxc.md).
	fmt.Printf("fastmlx serve (v%s) - mock decode backend\n", version)
	fmt.Printf("  listen      %s:%d\n", o.host, o.port)
	fmt.Printf("  base-path   %s\n", o.basePath)
	fmt.Printf("  model-dir   %s\n", o.modelDir)
	fmt.Printf("  max-conc    %d\n", o.maxConcurrentRequests)
	fmt.Printf("  sse-keep    %s\n", o.sseKeepaliveMode)
	if o.noCache {
		fmt.Println("  ssd-cache   disabled")
	} else {
		fmt.Printf("  ssd-cache   %s (max %s, hot %s)\n", o.pagedSSDCacheDir, o.pagedSSDCacheMaxSize, o.hotCacheMaxSize)
	}

	schedCfg := scheduler.DefaultConfig()
	schedCfg.MaxNumSeqs = o.maxConcurrentRequests
	schedCfg.EmbeddingBatchSize = o.embeddingBatchSize

	eng := enginecore.NewBatchedEngine(enginecore.Options{
		ModelName:     "mock-model",
		Scheduler:     schedCfg,
		MaxConcurrent: o.maxConcurrentRequests,
	})

	var apiKeys []string
	if o.apiKey != "" {
		apiKeys = []string{o.apiKey}
	}
	app := server.NewApp(server.Config{
		Addr:        net.JoinHostPort(o.host, strconv.Itoa(o.port)),
		Engine:      eng,
		APIKeys:     apiKeys,
		CORSOrigins: []string{"*"},
	})

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	return app.Run(ctx)
}

func runLaunch(args []string) error {
	return fmt.Errorf("launch lands in v0.8 (integrations)")
}

func runDiagnose(args []string) error {
	return fmt.Errorf("diagnose lands in v0.8")
}
