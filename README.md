# fastmlx

A continuous-batching, tiered-KV-cache MLX LLM inference server for Apple Silicon, written in Go: a
continuous-batching, tiered-KV-cache MLX LLM inference server for Apple Silicon, with an
OpenAI / Anthropic / Responses-compatible API, a menu-bar-managed admin dashboard, and KV cache that
persists across a hot RAM tier and a cold SSD tier.

## Goals

1. **Full feature parity** across the serving and management surface.
2. **2x faster on serving overhead and concurrent throughput.** Single-stream tok/s is *matched*
   (identical MLX Metal kernels); the win is in the GIL-free layers around the GPU - `net/http` +
   goroutines, zero-allocation SSE, concurrent cache I/O, and CPU work overlapped with the GPU step.
3. **A single static binary** providing `serve` / `launch` / `diagnose`, reading the
   same `~/.fastmlx/` config and cache layout.

The honest framing: a serving rewrite cannot beat MLX on raw single-stream tok/s (the GPU forward is
the same Metal kernels). The 2x lives in concurrent throughput, requests/sec, and time-to-first-token
under load - where Python's GIL serializes the scheduler, detokenizer, parsers, and SSE encoder.

## Status

Early. The build runs through milestones v0.1 to v1.0:

- **v0.1 (in progress)** - foundations: config/settings parsing, model discovery, CLI skeleton.
- v0.2 - `net/http` serving layer + scheduler behind a mock decode backend.
- v0.3 - tool/reasoning parsers, Anthropic + Responses routes, MCP.
- v0.4 - the compute backend: `mlxgo` cgo bindings over `mlx-c`, first real token.
- v0.5+ - tiered cache, speculative decoding, architecture breadth, management, multimodal, benchmark.

See `docs/` (TBD) and the spec for the full plan.

## Build

```
go build ./...      # serving + logic layers (no GPU required)
go test ./...
```

The compute backend (v0.4+) links against MLX via `mlx-c` and requires the Metal toolchain; a `nomlx`
build tag selects a pure-Go stub so everything builds and the serving layer tests without MLX.

## License

Licensed under the Apache License, Version 2.0.
