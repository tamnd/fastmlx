// SPDX-License-Identifier: MIT OR Apache-2.0

package compute

// Logits processors mirror mlx_lm.sample_utils.make_logits_processors, the
// factory the serving layer feeds with the OpenAI sampling params, so the
// numerics here are the parity target. Each processor takes the running token
// history and the current logits row and rewrites the row in place; the
// scheduler composes them per active sequence before sampling.
//
// This file is the pure, GPU-free half of the sampler stack: it is plain
// arithmetic over a []float32 logits row plus a []int token history, so it runs
// and is parity-tested on any host. The MLX seam is only the surrounding tensor
// plumbing (the logits come off an mlx_array, the row is one slice of a
// [batch, vocab] block); the penalty math itself is identical whether the row
// originates on the GPU or in this test harness.

// LogitsProcessor rewrites a single-sequence logits row in place, given the
// tokens generated so far. It matches the mlx-lm processor contract
// (tokens, logits) -> logits, specialized to one row.
type LogitsProcessor func(tokens []int, logits []float32)

// LogitsProcessorParams mirrors the keyword arguments of
// make_logits_processors. A nil penalty pointer means "not supplied"; a context
// size of zero or below means "use the whole history" (no truncation), matching
// a non-positive slice bound in the reference. LogitBias maps a token id to an
// additive bias and corresponds to the OpenAI logit_bias option.
type LogitsProcessorParams struct {
	LogitBias             map[int]float64
	RepetitionPenalty     *float64
	RepetitionContextSize int
	PresencePenalty       *float64
	PresenceContextSize   int
	FrequencyPenalty      *float64
	FrequencyContextSize  int
}

// DefaultPenaltyContextSize is the mlx-lm default window (20 tokens) for every
// penalty when the caller does not override it.
const DefaultPenaltyContextSize = 20

// window returns the trailing slice of tokens used by a penalty. A non-positive
// contextSize disables truncation, matching Python's tokens[-n:] with n<=0
// collapsing to "everything" only when the reference guards it; here the
// factory always passes the resolved positive default, so this just bounds the
// tail.
func window(tokens []int, contextSize int) []int {
	if contextSize > 0 && len(tokens) > contextSize {
		return tokens[len(tokens)-contextSize:]
	}
	return tokens
}

// MakeRepetitionPenalty ports make_repetition_penalty: a sign-aware
// multiplicative penalty applied once per unique token in the window. mlx
// gathers the selected logits from the original row and scatters the penalized
// values back, so duplicate tokens do not compound; the dedup here reproduces
// that. A logit below zero is multiplied by the penalty (pushed further down),
// otherwise it is divided. penalty == 1.0 is the numerical no-op the convention
// uses for "off".
func MakeRepetitionPenalty(penalty float64, contextSize int) LogitsProcessor {
	p := float32(penalty)
	return func(tokens []int, logits []float32) {
		if len(tokens) == 0 {
			return
		}
		seen := make(map[int]struct{})
		for _, t := range window(tokens, contextSize) {
			if t < 0 || t >= len(logits) {
				continue
			}
			if _, dup := seen[t]; dup {
				continue
			}
			seen[t] = struct{}{}
			if v := logits[t]; v < 0 {
				logits[t] = v * p
			} else {
				logits[t] = v / p
			}
		}
	}
}

// MakePresencePenalty ports make_presence_penalty: the OpenAI presence penalty,
// subtracting the penalty once from any token that occurs at least once in the
// window. mlx's logits[:, tokens] -= penalty is a gather/scatter assignment, so
// duplicate tokens are penalized only once; the dedup matches that.
func MakePresencePenalty(penalty float64, contextSize int) LogitsProcessor {
	p := float32(penalty)
	return func(tokens []int, logits []float32) {
		if len(tokens) == 0 {
			return
		}
		seen := make(map[int]struct{})
		for _, t := range window(tokens, contextSize) {
			if t < 0 || t >= len(logits) {
				continue
			}
			if _, dup := seen[t]; dup {
				continue
			}
			seen[t] = struct{}{}
			logits[t] -= p
		}
	}
}

// MakeFrequencyPenalty ports make_frequency_penalty: the OpenAI frequency
// penalty, subtracting the penalty for every occurrence of a token in the
// window. mlx uses the accumulating .at[:, tokens].subtract(penalty), so a token
// appearing k times is penalized k*penalty; no dedup here, by design.
func MakeFrequencyPenalty(penalty float64, contextSize int) LogitsProcessor {
	p := float32(penalty)
	return func(tokens []int, logits []float32) {
		if len(tokens) == 0 {
			return
		}
		for _, t := range window(tokens, contextSize) {
			if t < 0 || t >= len(logits) {
				continue
			}
			logits[t] -= p
		}
	}
}

// MakeLogitBias ports the logit_bias_processor: an additive bias per token id,
// independent of the token history. The keys come from a map so each id is
// applied once; addition is order-independent.
func MakeLogitBias(bias map[int]float64) LogitsProcessor {
	return func(_ []int, logits []float32) {
		for idx, val := range bias {
			if idx < 0 || idx >= len(logits) {
				continue
			}
			logits[idx] += float32(val)
		}
	}
}

// MakeLogitsProcessors assembles the processor chain in the same order as
// mlx_lm.sample_utils.make_logits_processors: logit_bias first, then the
// repetition, presence, and frequency penalties. A penalty is included only
// when its pointer is non-nil and the value is non-zero (the reference's
// `penalty is not None and penalty != 0`); a non-positive context size is
// resolved to the mlx-lm default of 20. logit_bias is skipped when empty.
func MakeLogitsProcessors(params LogitsProcessorParams) []LogitsProcessor {
	var procs []LogitsProcessor
	if len(params.LogitBias) > 0 {
		procs = append(procs, MakeLogitBias(params.LogitBias))
	}
	if params.RepetitionPenalty != nil && *params.RepetitionPenalty != 0 {
		procs = append(procs, MakeRepetitionPenalty(*params.RepetitionPenalty, resolveContext(params.RepetitionContextSize)))
	}
	if params.PresencePenalty != nil && *params.PresencePenalty != 0 {
		procs = append(procs, MakePresencePenalty(*params.PresencePenalty, resolveContext(params.PresenceContextSize)))
	}
	if params.FrequencyPenalty != nil && *params.FrequencyPenalty != 0 {
		procs = append(procs, MakeFrequencyPenalty(*params.FrequencyPenalty, resolveContext(params.FrequencyContextSize)))
	}
	return procs
}

// ApplyLogitsProcessors runs every processor over the row in order, in place.
func ApplyLogitsProcessors(procs []LogitsProcessor, tokens []int, logits []float32) {
	for _, p := range procs {
		p(tokens, logits)
	}
}

// resolveContext maps a non-positive (unset) context size to the mlx-lm default.
func resolveContext(n int) int {
	if n <= 0 {
		return DefaultPenaltyContextSize
	}
	return n
}
