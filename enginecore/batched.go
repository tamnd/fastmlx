// SPDX-License-Identifier: MIT OR Apache-2.0

package enginecore

import (
	"context"
	"strings"

	"github.com/tamnd/fastmlx/engine"
	"github.com/tamnd/fastmlx/pipeline"
	"github.com/tamnd/fastmlx/scheduler"
	"github.com/tamnd/fastmlx/tokenizer"
)

// BatchedEngine is the LLM engine: a tokenizer, a scheduler over a decode
// backend, and the core loop. Stage 2
// runs it over pipeline.MockDecode; the real compute.BatchGenerator drops in
// behind the same scheduler at the compute milestone.
type BatchedEngine struct {
	modelName string
	tok       tokenizer.Tokenizer
	core      *Core
	defaults  engine.SamplingParams
}

// Options configures a BatchedEngine.
type Options struct {
	ModelName     string
	Tokenizer     tokenizer.Tokenizer
	Decode        pipeline.DecodeStrategy
	Scheduler     scheduler.Config
	MaxConcurrent int
	Defaults      engine.SamplingParams
}

// NewBatchedEngine assembles the engine from its parts. If no tokenizer or decode
// backend is supplied, it wires the mock pair so the serving path runs.
func NewBatchedEngine(opts Options) *BatchedEngine {
	if opts.Tokenizer == nil {
		opts.Tokenizer = tokenizer.NewMock()
	}
	if opts.Decode == nil {
		opts.Decode = pipeline.NewMockDecode(opts.Tokenizer, "")
	}
	if opts.Scheduler.MaxNumSeqs == 0 {
		opts.Scheduler = scheduler.DefaultConfig()
	}
	sched := scheduler.New(opts.Scheduler, opts.Decode, opts.Tokenizer)
	return &BatchedEngine{
		modelName: opts.ModelName,
		tok:       opts.Tokenizer,
		core:      NewCore(sched, opts.MaxConcurrent),
		defaults:  opts.Defaults,
	}
}

// Start launches the engine loop.
func (e *BatchedEngine) Start(ctx context.Context) { e.core.Start(ctx) }

// Stop waits for the engine loop to drain and exit.
func (e *BatchedEngine) Stop() { e.core.Stop() }

// ModelName returns the served model identifier.
func (e *BatchedEngine) ModelName() string { return e.modelName }

// Tokenizer exposes the engine tokenizer.
func (e *BatchedEngine) Tokenizer() tokenizer.Tokenizer { return e.tok }

// Defaults returns the engine's base sampling parameters (lowest precedence in
// the sampling cascade).
func (e *BatchedEngine) Defaults() engine.SamplingParams { return e.defaults }

// InFlight reports admitted requests for /api/status.
func (e *BatchedEngine) InFlight() int { return e.core.InFlight() }

// Submit admits a request and returns its increment stream.
func (e *BatchedEngine) Submit(req *engine.Request) (<-chan engine.RequestOutput, error) {
	return e.core.Submit(req)
}

// Abort cancels an in-flight request.
func (e *BatchedEngine) Abort(id string) { e.core.Abort(id) }

// CountTokens returns the token count of text under the engine tokenizer.
func (e *BatchedEngine) CountTokens(text string) int { return len(e.tok.Encode(text)) }

// BuildPrompt renders chat messages into a single prompt string. The mock backend
// ignores prompt content beyond its token count, so this uses a simple, readable
// role-tagged format; the real Jinja chat-template renderer lands with the
// tokenizer milestone (spec 06).
func (e *BatchedEngine) BuildPrompt(msgs []engine.Message, tools []engine.Tool, opts engine.PromptOptions) (string, error) {
	var b strings.Builder
	for _, t := range tools {
		b.WriteString("<tool>")
		b.WriteString(t.Name)
		b.WriteString("</tool>\n")
	}
	for _, m := range msgs {
		b.WriteString(m.Role)
		b.WriteString(": ")
		b.WriteString(m.Content)
		b.WriteString("\n")
	}
	if opts.AddGenerationPrompt {
		b.WriteString("assistant: ")
	}
	return b.String(), nil
}
