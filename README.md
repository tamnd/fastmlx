# fastmlx

A continuous-batching, tiered-KV-cache MLX LLM inference server for Apple Silicon, written in Go.
It speaks the OpenAI, Anthropic, and Responses APIs, ships a menu-bar-managed admin dashboard, and
keeps a KV cache that spans a hot RAM tier and a cold SSD tier.

## Goals

1. **A complete serving and management surface.** Chat, completions, embeddings, reranking, tool and
   reasoning parsers, model discovery, and the admin dashboard all in one binary.
2. **2x faster on serving overhead and concurrent throughput.** Single-stream tok/s is *matched*
   (the same MLX Metal kernels do the forward pass); the win is in the layers around the GPU:
   `net/http` plus goroutines, zero-allocation SSE, concurrent cache I/O, and CPU work overlapped
   with the GPU step.
3. **A single static binary** providing `serve`, `launch`, and `diagnose`, reading a `~/.fastmlx/`
   config and cache layout.

The honest framing: a serving layer cannot beat MLX on raw single-stream tok/s, because the GPU
forward is the same Metal kernels either way. The 2x lives in concurrent throughput, requests per
second, and time-to-first-token under load, where a single-threaded Python server serializes the
scheduler, detokenizer, parsers, and SSE encoder.

## Status

Early. The build runs through milestones v0.1 to v1.0:

- **v0.1 (done)** - foundations: config and settings parsing, model discovery, CLI skeleton.
- **v0.2 (done)** - `net/http` serving layer plus scheduler behind a mock decode backend. The
  OpenAI-compatible chat, completions, models, and health path runs end to end with
  zero-allocation SSE.
- v0.3 - tool and reasoning parsers, Anthropic and Responses routes, MCP.
- v0.4 - the compute backend: cgo bindings over `mlx-c`, first real token.
- v0.5+ - tiered cache, speculative decoding, architecture breadth, management, multimodal, benchmark.

Each milestone lands as its own pull request carrying its code, tests, and benchmarks.

## Build

```
go build ./...      # serving and logic layers (no GPU required)
go test ./...
```

The compute backend (v0.4+) links against MLX via `mlx-c` and requires the Metal toolchain. A `nomlx`
build tag selects a pure-Go stub so everything builds and the serving layer tests run without MLX.

## License

Licensed under either of

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option. Both licenses are offered so you can pick whichever fits your project. Unless you
state otherwise, any contribution you intentionally submit for inclusion in this work shall be dual
licensed as above, without any additional terms or conditions.
