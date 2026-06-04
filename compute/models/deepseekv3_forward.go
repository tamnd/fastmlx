// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"fmt"
	"math"

	"github.com/tamnd/fastmlx/mlxgo"
)

// mlaLayernormEps is the fixed epsilon the q_a and kv_a RMSNorms use. The
// reference hard-codes 1e-6 on these two low-rank layernorms regardless of the
// model's rms_norm_eps, which the block layernorms read.
const mlaLayernormEps = 1e-6

// rope dispatch kinds for the rotary key band.
const (
	ropePlain  = iota // traditional rope, no scaling
	ropeLinear        // traditional rope with a 1/factor position scale
	ropeYarn          // yarn: a host-built frequency table plus an mscale pre-scale
)

// deepseekV3Layer holds one decoder block's weights. The attention is multi-head
// latent attention, so it carries either the plain q_proj or the low-rank
// q_a/q_b pair, the compressed kv_a projection and its layernorm, and the
// absorbed embed_q and unembed_out projections. The MLP is either a dense
// gate/up/down or the routed mixture: the router gate weight and correction
// bias, the three stacked switch_mlp tensors, and an optional shared expert.
type deepseekV3Layer struct {
	inputLayernorm         *mlxgo.Array
	postAttentionLayernorm *mlxgo.Array

	qProj                       *mlxgo.Array // plain query path (nil when q_lora is on)
	qAProj, qALayernorm, qBProj *mlxgo.Array // low-rank query path
	qAProjBias                  *mlxgo.Array // optional
	kvAProj, kvAProjBias        *mlxgo.Array
	kvALayernorm                *mlxgo.Array
	embedQ, unembedOut          *mlxgo.Array
	oProj, oProjBias            *mlxgo.Array

	isMoE                      bool
	gateProj, upProj, downProj *mlxgo.Array // dense MLP

	gateW, eScoreBias                *mlxgo.Array // router
	switchGate, switchUp, switchDown *mlxgo.Array // stacked experts
	sharedGate, sharedUp, sharedDown *mlxgo.Array // optional shared expert
}

// DeepseekV3Model is an assembled DeepSeek-V3 model: the decoded args, the
// weight tensors wired into typed fields, and the precomputed rope schedule for
// the rotary key band. The weights arrive as the loader's name-to-array map (the
// keys are exactly DeepseekV3Args.WeightNames after the pre-load patch).
type DeepseekV3Model struct {
	args        *DeepseekV3Args
	embedTokens *mlxgo.Array
	layers      []deepseekV3Layer
	norm        *mlxgo.Array
	lmHead      *mlxgo.Array

	ropeKind   int
	ropeLinear float32      // 1/factor for the linear schedule
	yarnFreqs  *mlxgo.Array // host-built yarn frequency table
	yarnMScale float32      // yarn magnitude pre-scale (1 means none)
}

// NewDeepseekV3Model wires a sanitized weight map into a runnable model. Every
// key DeepseekV3Args.WeightNames reports for a given layer must be present; the
// attention biases load only when attention_bias is set, and the MLP keys follow
// the dense-or-routed split. The rope schedule is built once from rope_scaling.
func NewDeepseekV3Model(args *DeepseekV3Args, weights map[string]*mlxgo.Array) (*DeepseekV3Model, error) {
	get := func(name string) (*mlxgo.Array, error) {
		w, ok := weights[name]
		if !ok || w == nil {
			return nil, fmt.Errorf("deepseekv3: missing weight %q", name)
		}
		return w, nil
	}
	opt := func(name string) *mlxgo.Array { return weights[name] }

	m := &DeepseekV3Model{args: args, layers: make([]deepseekV3Layer, args.NumLayers())}
	var err error
	if m.embedTokens, err = get("model.embed_tokens.weight"); err != nil {
		return nil, err
	}
	if m.norm, err = get("model.norm.weight"); err != nil {
		return nil, err
	}
	if m.lmHead, err = get("lm_head.weight"); err != nil {
		return nil, err
	}

	for i := range m.layers {
		p := fmt.Sprintf("model.layers.%d.", i)
		ly := &m.layers[i]
		if ly.inputLayernorm, err = get(p + "input_layernorm.weight"); err != nil {
			return nil, err
		}
		if ly.postAttentionLayernorm, err = get(p + "post_attention_layernorm.weight"); err != nil {
			return nil, err
		}

		ap := p + "self_attn."
		if args.HasQLora() {
			if ly.qAProj, err = get(ap + "q_a_proj.weight"); err != nil {
				return nil, err
			}
			if ly.qALayernorm, err = get(ap + "q_a_layernorm.weight"); err != nil {
				return nil, err
			}
			if ly.qBProj, err = get(ap + "q_b_proj.weight"); err != nil {
				return nil, err
			}
			if args.AttentionBias {
				ly.qAProjBias = opt(ap + "q_a_proj.bias")
			}
		} else if ly.qProj, err = get(ap + "q_proj.weight"); err != nil {
			return nil, err
		}
		if ly.kvAProj, err = get(ap + "kv_a_proj_with_mqa.weight"); err != nil {
			return nil, err
		}
		if ly.kvALayernorm, err = get(ap + "kv_a_layernorm.weight"); err != nil {
			return nil, err
		}
		if ly.embedQ, err = get(ap + "embed_q.weight"); err != nil {
			return nil, err
		}
		if ly.unembedOut, err = get(ap + "unembed_out.weight"); err != nil {
			return nil, err
		}
		if ly.oProj, err = get(ap + "o_proj.weight"); err != nil {
			return nil, err
		}
		if args.AttentionBias {
			ly.kvAProjBias = opt(ap + "kv_a_proj_with_mqa.bias")
			ly.oProjBias = opt(ap + "o_proj.bias")
		}

		mp := p + "mlp."
		if args.IsMoELayer(i) {
			ly.isMoE = true
			if ly.gateW, err = get(mp + "gate.weight"); err != nil {
				return nil, err
			}
			if ly.eScoreBias, err = get(mp + "gate.e_score_correction_bias"); err != nil {
				return nil, err
			}
			if ly.switchGate, err = get(mp + "switch_mlp.gate_proj.weight"); err != nil {
				return nil, err
			}
			if ly.switchUp, err = get(mp + "switch_mlp.up_proj.weight"); err != nil {
				return nil, err
			}
			if ly.switchDown, err = get(mp + "switch_mlp.down_proj.weight"); err != nil {
				return nil, err
			}
			if args.HasSharedExperts() {
				if ly.sharedGate, err = get(mp + "shared_experts.gate_proj.weight"); err != nil {
					return nil, err
				}
				if ly.sharedUp, err = get(mp + "shared_experts.up_proj.weight"); err != nil {
					return nil, err
				}
				if ly.sharedDown, err = get(mp + "shared_experts.down_proj.weight"); err != nil {
					return nil, err
				}
			}
		} else {
			if ly.gateProj, err = get(mp + "gate_proj.weight"); err != nil {
				return nil, err
			}
			if ly.upProj, err = get(mp + "up_proj.weight"); err != nil {
				return nil, err
			}
			if ly.downProj, err = get(mp + "down_proj.weight"); err != nil {
				return nil, err
			}
		}
	}

	m.ropeKind = ropePlain
	if args.RopeScaling != nil {
		switch args.RopeScaling.Type {
		case "", "default":
			m.ropeKind = ropePlain
		case "linear":
			m.ropeKind = ropeLinear
			m.ropeLinear = float32(1.0 / args.RopeScaling.Factor)
		case "yarn", "deepseek_yarn", "telechat3-yarn":
			m.ropeKind = ropeYarn
			freqs, mscale := deepseekYarnFreqs(args)
			if m.yarnFreqs, err = mlxgo.NewFloat32(freqs, len(freqs)); err != nil {
				return nil, err
			}
			m.yarnMScale = float32(mscale)
		default:
			return nil, fmt.Errorf("deepseekv3: unsupported rope type %q", args.RopeScaling.Type)
		}
	}
	return m, nil
}

// Forward runs a single sequence of tokens through the model and returns the
// logits, shaped [1, len(tokens), vocab_size]. caches holds one KVTensorCache
// per layer; the latent attention stores the compressed kv and the rotary key
// band as the cache's key and value tensors. The whole prompt arrives as one
// call (L>1, the prefill path) and then single tokens (L==1, the decode path),
// so both branches of the MLA absorption run.
func (m *DeepseekV3Model) Forward(tokens []int32, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	if len(caches) != len(m.layers) {
		return nil, fmt.Errorf("deepseekv3: got %d caches, want %d", len(caches), len(m.layers))
	}
	a := m.args
	L := len(tokens)
	eps := float32(a.RMSNormEps)
	scale := float32(a.AttentionScale())
	nh := a.NumAttentionHeads
	qNope := a.QKNopeHeadDim
	qRope := a.QKRopeHeadDim
	qHead := a.QHeadDim()
	kvLora := a.KVLoraRank
	vHead := a.VHeadDim

	b := &fb{s: s}

	idx, err := mlxgo.NewInt32(tokens, L)
	if err != nil {
		return nil, err
	}
	h, err := mlxgo.Take(m.embedTokens, idx, 0, s)
	if err != nil {
		return nil, err
	}
	h = b.reshape(h, []int{1, L, a.HiddenSize})

	for i := range m.layers {
		layer := &m.layers[i]
		cache := caches[i]
		offset := cache.Offset

		// Multi-head latent attention.
		x := b.rmsNorm(h, layer.inputLayernorm, eps)
		var q *mlxgo.Array
		if a.HasQLora() {
			qa := b.linearBias(x, layer.qAProj, layer.qAProjBias)
			qa = b.rmsNorm(qa, layer.qALayernorm, mlaLayernormEps)
			q = b.linear(qa, layer.qBProj)
		} else {
			q = b.linear(x, layer.qProj)
		}
		q = b.transpose(b.reshape(q, []int{1, L, nh, qHead}), []int{0, 2, 1, 3})
		qParts := b.splitLast(q, []int{qNope})
		qn, qpe := qParts[0], qParts[1]

		compressed := b.linearBias(x, layer.kvAProj, layer.kvAProjBias)
		cParts := b.splitLast(compressed, []int{kvLora})
		comp, kpe := cParts[0], cParts[1]
		kpe = b.transpose(b.reshape(kpe, []int{1, L, 1, qRope}), []int{0, 2, 1, 3})
		kvLatent := b.rmsNorm(comp, layer.kvALayernorm, mlaLayernormEps)
		kvLatent = b.reshape(kvLatent, []int{1, 1, L, kvLora})

		qpe = m.applyRope(b, qpe, offset)
		kpe = m.applyRope(b, kpe, offset)
		if b.err == nil {
			kvLatent, kpe, b.err = cache.Update(kvLatent, kpe, s)
		}

		// pe_scores carries the rotary band contribution and, for the prefill,
		// the causal mask. It becomes the additive mask the SDPA softmax adds to
		// the latent (nope-band) scores, so both bands share the one scale.
		peScores := b.matmul(b.scalarMul(qpe, scale), b.transpose(kpe, []int{0, 1, 3, 2}))
		if L > 1 {
			peScores = b.add(peScores, m.causalMask(b, L, offset))
		}

		var out *mlxgo.Array
		if L == 1 {
			qn = b.multiLinear(qn, layer.embedQ, true)
			out = b.sdpaWith(qn, kvLatent, kvLatent, scale, "", peScores)
			out = b.multiLinear(out, layer.unembedOut, true)
		} else {
			kk := b.multiLinear(kvLatent, layer.embedQ, false)
			vv := b.multiLinear(kvLatent, layer.unembedOut, true)
			out = b.sdpaWith(qn, kk, vv, scale, "", peScores)
		}
		out = b.reshape(b.transpose(out, []int{0, 2, 1, 3}), []int{1, L, nh * vHead})
		out = b.linearBias(out, layer.oProj, layer.oProjBias)
		h = b.add(h, out)

		// Dense or routed MLP.
		y := b.rmsNorm(h, layer.postAttentionLayernorm, eps)
		if layer.isMoE {
			y = b.deepseekMoE(y, layer, a, L)
		} else {
			y = b.deepseekMLP(y, layer.gateProj, layer.upProj, layer.downProj)
		}
		h = b.add(h, y)
	}

	h = b.rmsNorm(h, m.norm, eps)
	logits := b.linear(h, m.lmHead)
	if b.err != nil {
		return nil, b.err
	}
	return logits, nil
}

// applyRope rotates the rotary key band x with the model's configured schedule.
// Plain and linear schedules are traditional rope with a position scale; the
// yarn schedule pre-scales by mscale then rotates from the host-built table.
func (m *DeepseekV3Model) applyRope(b *fb, x *mlxgo.Array, offset int) *mlxgo.Array {
	dims := m.args.QKRopeHeadDim
	theta := float32(m.args.RopeTheta)
	switch m.ropeKind {
	case ropeLinear:
		return b.ropeScaled(x, dims, true, theta, m.ropeLinear, offset)
	case ropeYarn:
		if m.yarnMScale != 1 {
			x = b.scalarMul(x, m.yarnMScale)
		}
		return b.ropeFreqsTrad(x, dims, true, offset, m.yarnFreqs)
	default:
		return b.ropeTrad(x, dims, true, theta, offset)
	}
}

// causalMask builds the additive prefill mask shaped [1, 1, L, offset+L]: zero
// where a query may attend (key position at or before the query's global
// position) and a large negative elsewhere, so the softmax drops future keys.
func (m *DeepseekV3Model) causalMask(b *fb, L, offset int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	S := offset + L
	arr, err := mlxgo.NewFloat32(causalAdditiveMask(L, offset), 1, 1, L, S)
	if err != nil {
		b.err = err
		return nil
	}
	return arr
}

// deepseekMLP is the dense SwiGLU MLP: down(silu(gate(x)) * up(x)).
func (b *fb) deepseekMLP(x, gateW, upW, downW *mlxgo.Array) *mlxgo.Array {
	gate := b.silu(b.linear(x, gateW))
	up := b.linear(x, upW)
	return b.linear(b.mul(gate, up), downW)
}

// deepseekMoE runs the routed mixture: the router picks the per-token experts
// and weights, the SwitchGLU runs the expert MLPs, the outputs are weighted and
// summed over the selection, and the optional shared expert runs on every token.
func (b *fb) deepseekMoE(x *mlxgo.Array, layer *deepseekV3Layer, a *DeepseekV3Args, L int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	gates := b.linear(x, layer.gateW)
	inds, weights := b.routeExperts(gates, layer.eScoreBias, a, L)
	y := b.switchGLU(x, layer.switchGate, layer.switchUp, layer.switchDown, inds)
	// (y * scores[..., None]).sum(axis=-2): weight each expert output, then sum
	// over the top_k axis back to one hidden vector per token.
	wr := b.reshape(weights, []int{1, L, a.NumExpertsPerTok, 1})
	y = b.sumAxis(b.mul(y, wr), 2, false)
	if layer.sharedGate != nil {
		y = b.add(y, b.deepseekMLP(x, layer.sharedGate, layer.sharedUp, layer.sharedDown))
	}
	return y
}

// routeExperts is the GPU port of the reference group_expert_select. It scores
// the experts with a sigmoid, adds the correction bias, restricts the choice to
// the topk_group highest-scoring groups (a group scores by its two best biased
// experts), takes the top_k experts by biased score, gathers the original
// pre-bias scores at those experts, optionally normalizes across the selection,
// and scales by routed_scaling_factor. The routing stays on the device so the
// forward never reads back mid-graph. gates is [1, L, n_routed_experts]; the
// returned indices and weights are [1, L, num_experts_per_tok].
func (b *fb) routeExperts(gates, bias *mlxgo.Array, a *DeepseekV3Args, L int) (inds, weights *mlxgo.Array) {
	if b.err != nil {
		return nil, nil
	}
	topK := a.NumExpertsPerTok
	nGroup := a.NGroup
	topkGroup := a.TopkGroup
	E := a.NRoutedExperts
	const lastAxis = 2

	scores := b.sigmoidArr(gates)
	orig := scores
	scores = b.add(scores, bias)

	if nGroup > 1 {
		per := E / nGroup
		const groupAxis, innerAxis = 2, 3
		sc := b.reshape(scores, []int{1, L, nGroup, per})
		// group score = sum of the two best biased experts in the group.
		top2 := b.takeAlongAxis(sc, b.sliceFirst(b.argpartition(b.scalarMul(sc, -1), 1, innerAxis), 2, innerAxis), innerAxis)
		groupScores := b.sumAxis(top2, innerAxis, true)
		// zero the n_group - topk_group lowest-scoring groups.
		if k := nGroup - topkGroup; k > 0 {
			groupIdx := b.sliceFirst(b.argpartition(groupScores, k-1, groupAxis), k, groupAxis)
			sc = b.putAlongAxis(sc, groupIdx, b.scalar(0), groupAxis)
		}
		scores = b.reshape(sc, []int{1, L, E})
	}

	inds = b.sliceFirst(b.argpartition(b.scalarMul(scores, -1), topK-1, lastAxis), topK, lastAxis)
	weights = b.takeAlongAxis(orig, inds, lastAxis)
	if topK > 1 && a.NormTopkProb {
		weights = b.div(weights, b.sumAxis(weights, lastAxis, true))
	}
	weights = b.scalarMul(weights, float32(a.RoutedScalingFactor))
	return inds, weights
}

// multiLinear is the absorbed MLA projection: a batched per-head matmul against
// a weight stored [num_heads, output_dims, input_dims]. transpose multiplies by
// the swapped weight (the default x @ W^T), and the non-transposed form (x @ W)
// is the prefill key path, which reads the latent through the same heads.
func (b *fb) multiLinear(x, w *mlxgo.Array, transpose bool) *mlxgo.Array {
	if transpose {
		return b.matmul(x, b.transpose(w, []int{0, 2, 1}))
	}
	return b.matmul(x, w)
}

// splitLast splits x along its last axis at the given section boundaries.
func (b *fb) splitLast(x *mlxgo.Array, indices []int) []*mlxgo.Array {
	if b.err != nil {
		return make([]*mlxgo.Array, len(indices)+1)
	}
	parts, err := mlxgo.SplitSections(x, indices, -1, b.s)
	if err != nil {
		b.err = err
		return make([]*mlxgo.Array, len(indices)+1)
	}
	return parts
}

// sliceFirst returns the first n slices of x along axis.
func (b *fb) sliceFirst(x *mlxgo.Array, n, axis int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	parts, err := mlxgo.SplitSections(x, []int{n}, axis, b.s)
	if err != nil {
		b.err = err
		return nil
	}
	return parts[0]
}

func (b *fb) sigmoidArr(x *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Sigmoid(x, b.s)
	b.err = err
	return r
}

func (b *fb) argpartition(x *mlxgo.Array, kth, axis int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Argpartition(x, kth, axis, b.s)
	b.err = err
	return r
}

func (b *fb) takeAlongAxis(x, indices *mlxgo.Array, axis int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.TakeAlongAxis(x, indices, axis, b.s)
	b.err = err
	return r
}

func (b *fb) putAlongAxis(x, indices, values *mlxgo.Array, axis int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.PutAlongAxis(x, indices, values, axis, b.s)
	b.err = err
	return r
}

func (b *fb) sumAxis(x *mlxgo.Array, axis int, keepDims bool) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Sum(x, axis, keepDims, b.s)
	b.err = err
	return r
}

func (b *fb) div(x, y *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Div(x, y, b.s)
	b.err = err
	return r
}

// ropeFreqsTrad applies rotary embedding from an explicit frequency table with
// the traditional (interleaved) pairing DeepSeek's yarn schedule uses.
func (b *fb) ropeFreqsTrad(x *mlxgo.Array, dims int, traditional bool, offset int, freqs *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.RoPEWithFreqs(x, dims, traditional, 1, offset, freqs, b.s)
	b.err = err
	return r
}

// causalAdditiveMask builds the row-major [L, offset+L] additive causal mask:
// zero where key position j is at or before the query's global position
// offset+i, and a large negative value elsewhere.
func causalAdditiveMask(L, offset int) []float32 {
	S := offset + L
	m := make([]float32, L*S)
	const negInf = -1e30
	for i := range L {
		qpos := offset + i
		row := i * S
		for j := qpos + 1; j < S; j++ {
			m[row+j] = negInf
		}
	}
	return m
}

// deepseekYarnFreqs builds the yarn rotary frequency table and the magnitude
// pre-scale on the host, mirroring the reference YarnRoPE. The table has
// qk_rope_head_dim/2 entries; the pre-scale multiplies the rotary band before
// the rotation when it differs from one.
func deepseekYarnFreqs(a *DeepseekV3Args) ([]float32, float64) {
	rs := a.RopeScaling
	dims := a.QKRopeHeadDim
	base := a.RopeTheta
	sf := rs.Factor
	origMax := float64(rs.OriginalMaxPositionEmbeddings)
	if origMax <= 0 {
		origMax = 4096
	}
	betaFast := rs.BetaFast
	if betaFast == 0 {
		betaFast = 32
	}
	betaSlow := rs.BetaSlow
	if betaSlow == 0 {
		betaSlow = 1
	}
	const mscalePlain = 1.0
	mscaleAllDim := rs.MScaleAllDim

	findDim := func(numRot float64) float64 {
		return float64(dims) * math.Log(origMax/(numRot*2*math.Pi)) / (2 * math.Log(base))
	}
	low := math.Floor(findDim(betaFast))
	high := math.Ceil(findDim(betaSlow))
	if low < 0 {
		low = 0
	}
	if high > float64(dims-1) {
		high = float64(dims - 1)
	}
	if low == high {
		high += 0.001
	}

	getMscale := func(scale, ms float64) float64 {
		if scale <= 1 {
			return 1.0
		}
		return 0.1*ms*math.Log(scale) + 1.0
	}
	mscale := getMscale(sf, mscalePlain) / getMscale(sf, mscaleAllDim)

	half := dims / 2
	freqs := make([]float32, half)
	for j := range half {
		freqExtra := math.Pow(base, float64(2*j)/float64(dims))
		freqInter := sf * freqExtra
		ramp := (float64(j) - low) / (high - low)
		if ramp < 0 {
			ramp = 0
		}
		if ramp > 1 {
			ramp = 1
		}
		freqMask := 1.0 - ramp
		freqs[j] = float32((freqInter * freqExtra) / (freqInter*freqMask + freqExtra*(1-freqMask)))
	}
	return freqs, mscale
}
