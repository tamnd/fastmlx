// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"fmt"
	"math"

	"github.com/tamnd/fastmlx/mlxgo"
)

// gatedDeltaEps is the fixed epsilon the gated delta net applies in its two
// weightless query/key RMS norms and the gated output norm. The reference hard
// codes 1e-6 there regardless of rms_norm_eps.
const gatedDeltaEps = 1e-6

// qwen3NextLinear holds one gated delta net's weight tensors (a recurrent,
// linear-attention token mixer with a depthwise causal convolution).
type qwen3NextLinear struct {
	aLog       *mlxgo.Array // per value head log-decay, shape [num_v_heads]
	conv1d     *mlxgo.Array // depthwise conv weight, shape [conv_dim, kernel, 1]
	dtBias     *mlxgo.Array // per value head softplus bias, shape [num_v_heads]
	inProjBA   *mlxgo.Array // fused beta/alpha projection
	inProjQKVZ *mlxgo.Array // fused q/k/v/gate projection
	norm       *mlxgo.Array // gated output RMS norm weight, shape [head_v_dim]
	outProj    *mlxgo.Array
}

// qwen3NextAttn holds one full-attention block's weights. The query projection is
// double width: half is the query, half is the output gate.
type qwen3NextAttn struct {
	qProj, kProj, vProj, oProj *mlxgo.Array
	qNorm, kNorm               *mlxgo.Array
	qBias, kBias, vBias, oBias *mlxgo.Array // nil unless attention_bias
}

// qwen3NextMLP holds one block's feed-forward weights: either a dense SwiGLU or a
// sparse mixture with a shared expert, chosen by isMoE.
type qwen3NextMLP struct {
	isMoE                      bool
	gateProj, upProj, downProj *mlxgo.Array // dense SwiGLU
	gate                       *mlxgo.Array // router
	switchGate, switchUp       *mlxgo.Array // stacked experts
	switchDown                 *mlxgo.Array
	sharedGate, sharedUp       *mlxgo.Array // shared expert SwiGLU
	sharedDown                 *mlxgo.Array
	sharedExpertGate           *mlxgo.Array // shared expert sigmoid gate
}

// qwen3NextLayer is one decoder block: the two layernorms, one of the two token
// mixers (linear or full attention), and one MLP.
type qwen3NextLayer struct {
	inputLayernorm         *mlxgo.Array
	postAttentionLayernorm *mlxgo.Array
	isLinear               bool
	linear                 qwen3NextLinear
	attn                   qwen3NextAttn
	mlp                    qwen3NextMLP
}

// Qwen3NextModel is an assembled Qwen3-Next hybrid model: the decoded args plus
// the per-layer weights wired into typed fields.
type Qwen3NextModel struct {
	args        *Qwen3NextArgs
	embedTokens *mlxgo.Array
	layers      []qwen3NextLayer
	norm        *mlxgo.Array
	lmHead      *mlxgo.Array // nil when the head is tied to the embedding table
}

// weightField pairs a weight-map key with the model field it loads into, so the
// constructor can list a layer's tensors and pull them in one loop.
type weightField struct {
	name string
	dst  **mlxgo.Array
}

// NewQwen3NextModel wires a sanitized weight map into a runnable model. Every key
// in Qwen3NextArgs.WeightNames must be present; the attention biases are pulled
// only when attention_bias is set. A non-nil rope_scaling is not yet supported by
// the forward and is rejected here so a checkpoint that needs it fails at load
// rather than producing wrong numbers.
func NewQwen3NextModel(args *Qwen3NextArgs, weights map[string]*mlxgo.Array) (*Qwen3NextModel, error) {
	if args.RopeScaling != nil {
		return nil, fmt.Errorf("qwen3next: rope_scaling is not supported yet")
	}
	get := func(name string) (*mlxgo.Array, error) {
		w, ok := weights[name]
		if !ok || w == nil {
			return nil, fmt.Errorf("qwen3next: missing weight %q", name)
		}
		return w, nil
	}
	m := &Qwen3NextModel{args: args, layers: make([]qwen3NextLayer, args.NumLayers())}
	var err error
	if m.embedTokens, err = get("model.embed_tokens.weight"); err != nil {
		return nil, err
	}
	if m.norm, err = get("model.norm.weight"); err != nil {
		return nil, err
	}
	for i := range m.layers {
		layer := &m.layers[i]
		p := fmt.Sprintf("model.layers.%d.", i)
		fields := []weightField{
			{p + "input_layernorm.weight", &layer.inputLayernorm},
			{p + "post_attention_layernorm.weight", &layer.postAttentionLayernorm},
		}
		layer.isLinear = args.IsLinear(i)
		if layer.isLinear {
			lp := p + "linear_attn."
			fields = append(fields,
				weightField{lp + "A_log", &layer.linear.aLog},
				weightField{lp + "conv1d.weight", &layer.linear.conv1d},
				weightField{lp + "dt_bias", &layer.linear.dtBias},
				weightField{lp + "in_proj_ba.weight", &layer.linear.inProjBA},
				weightField{lp + "in_proj_qkvz.weight", &layer.linear.inProjQKVZ},
				weightField{lp + "norm.weight", &layer.linear.norm},
				weightField{lp + "out_proj.weight", &layer.linear.outProj},
			)
		} else {
			ap := p + "self_attn."
			fields = append(fields,
				weightField{ap + "q_proj.weight", &layer.attn.qProj},
				weightField{ap + "k_proj.weight", &layer.attn.kProj},
				weightField{ap + "v_proj.weight", &layer.attn.vProj},
				weightField{ap + "o_proj.weight", &layer.attn.oProj},
				weightField{ap + "q_norm.weight", &layer.attn.qNorm},
				weightField{ap + "k_norm.weight", &layer.attn.kNorm},
			)
			if args.AttentionBias {
				layer.attn.qBias = weights[ap+"q_proj.bias"]
				layer.attn.kBias = weights[ap+"k_proj.bias"]
				layer.attn.vBias = weights[ap+"v_proj.bias"]
				layer.attn.oBias = weights[ap+"o_proj.bias"]
			}
		}
		mp := p + "mlp."
		if args.IsMoELayer(i) {
			layer.mlp.isMoE = true
			fields = append(fields,
				weightField{mp + "gate.weight", &layer.mlp.gate},
				weightField{mp + "switch_mlp.gate_proj.weight", &layer.mlp.switchGate},
				weightField{mp + "switch_mlp.up_proj.weight", &layer.mlp.switchUp},
				weightField{mp + "switch_mlp.down_proj.weight", &layer.mlp.switchDown},
				weightField{mp + "shared_expert.gate_proj.weight", &layer.mlp.sharedGate},
				weightField{mp + "shared_expert.up_proj.weight", &layer.mlp.sharedUp},
				weightField{mp + "shared_expert.down_proj.weight", &layer.mlp.sharedDown},
				weightField{mp + "shared_expert_gate.weight", &layer.mlp.sharedExpertGate},
			)
		} else {
			fields = append(fields,
				weightField{mp + "gate_proj.weight", &layer.mlp.gateProj},
				weightField{mp + "up_proj.weight", &layer.mlp.upProj},
				weightField{mp + "down_proj.weight", &layer.mlp.downProj},
			)
		}
		for _, f := range fields {
			if *f.dst, err = get(f.name); err != nil {
				return nil, err
			}
		}
	}
	if args.TieWordEmbeddings {
		m.lmHead = nil
	} else if m.lmHead, err = get("lm_head.weight"); err != nil {
		return nil, err
	}
	return m, nil
}

// Forward runs one sequence of tokens through the model and returns the logits,
// shaped [1, len(tokens), vocab_size]. caches holds one KVTensorCache per layer;
// a linear layer reads and writes its convolution window and recurrent state, an
// attention layer grows its key and value tensors. The first kernel op is the
// embedding gather, so under the stub build Forward returns ErrMLXUnavailable
// there, which confirms the wiring on a host without MLX.
func (m *Qwen3NextModel) Forward(tokens []int32, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	return m.forwardBL(tokens, 1, len(tokens), caches, s)
}

// BatchDecode runs one decode step for a synchronized batch of sequences and
// returns the logits, shaped [batch, 1, vocab_size]. tokens is the row-major
// [batch, 1] block of one token per row. The rows share an offset, so the L==1
// step adds no attention mask. The hybrid stack handles the batch the same way
// the dense models do, with one extra subtlety: a linear (gated delta net) layer
// carries a recurrent state and a convolution window, which now lead with the
// batch axis ([batch, num_v_heads, v_dim, k_dim] and [batch, kernel-1, conv_dim]),
// and the one-timestep recurrence advances every row in parallel by broadcasting
// over that axis. With L==1 the per-step loop runs once, so each row takes exactly
// one recurrent step, the decode the throughput path needs.
func (m *Qwen3NextModel) BatchDecode(tokens []int32, batch int, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	return m.forwardBL(tokens, batch, 1, caches, s)
}

// forwardBL is the shared body: batch rows of L tokens each (row-major, batch*L
// values) through the hybrid decoder, returning [batch, L, vocab_size]. Forward
// calls it with batch 1 and L the prompt length, BatchDecode with L 1 and the
// batch width.
func (m *Qwen3NextModel) forwardBL(tokens []int32, batch, L int, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	if len(caches) != len(m.layers) {
		return nil, fmt.Errorf("qwen3next: got %d caches, want %d", len(caches), len(m.layers))
	}
	a := m.args
	eps := float32(a.RMSNormEps)
	b := &fb{s: s}

	idx, err := mlxgo.NewInt32(tokens, batch*L)
	if err != nil {
		return nil, err
	}
	h, err := mlxgo.Take(m.embedTokens, idx, 0, s)
	if err != nil {
		return nil, err
	}
	h = b.reshape(h, []int{batch, L, a.HiddenSize})

	// The full-attention layers read their mask and per-row rope offset from the
	// batch-aware cache, the same as the dense forwards; the linear (gated delta)
	// layers take the cohort's per-row left padding so a ragged prefill can drop the
	// padding positions from the convolution and the recurrence. Both fold to the
	// uniform fast path when the cohort is not left-padded. The owning offset is
	// uniform across layers, so all three are read once from caches[0].
	mode, mask, err := caches[0].AttnMask(batch, L, s)
	if err != nil {
		return nil, err
	}
	ropeOff := caches[0].RopeOffsets()
	leftPad := caches[0].LeftPad()

	for i := range m.layers {
		layer := &m.layers[i]
		cache := caches[i]

		x := b.rmsNorm(h, layer.inputLayernorm, eps)
		var r *mlxgo.Array
		if layer.isLinear {
			r = b.gatedDeltaNet(x, &layer.linear, a, cache, leftPad, batch, L)
		} else {
			r = b.qwen3NextAttention(x, &layer.attn, a, cache, mode, mask, ropeOff, batch, L)
		}
		h = b.add(h, r)

		y := b.rmsNorm(h, layer.postAttentionLayernorm, eps)
		if layer.mlp.isMoE {
			y = b.qwen3NextMoE(y, &layer.mlp, a, batch, L)
		} else {
			y = b.swiglu(y, layer.mlp.gateProj, layer.mlp.upProj, layer.mlp.downProj)
		}
		h = b.add(h, y)
	}

	h = b.rmsNorm(h, m.norm, eps)
	head := m.lmHead
	if head == nil {
		head = m.embedTokens
	}
	logits := b.linear(h, head)
	if b.err != nil {
		return nil, b.err
	}
	return logits, nil
}

// qwen3NextAttention is the gated full-attention block. The query projection is
// split into the query proper and an output gate; the gate multiplies the
// attention result (after a sigmoid) before the output projection.
func (b *fb) qwen3NextAttention(x *mlxgo.Array, attn *qwen3NextAttn, a *Qwen3NextArgs, cache *KVTensorCache, mode string, mask *mlxgo.Array, ropeOff []int, batch, L int) *mlxgo.Array {
	nh := a.NumAttentionHeads
	nkv := a.NumKeyValueHeads
	hd := a.HeadDim
	eps := float32(a.RMSNormEps)
	theta := float32(a.RopeTheta)
	scale := float32(a.AttentionScale())
	ropeDims := a.RopeDims()
	offset := cache.Offset

	qpo := b.linearBias(x, attn.qProj, attn.qBias)
	qpo = b.reshape(qpo, []int{batch, L, nh, 2 * hd})
	parts := b.splitLast(qpo, []int{hd})
	queries := parts[0]
	gate := b.reshape(parts[1], []int{batch, L, nh * hd})

	keys := b.linearBias(x, attn.kProj, attn.kBias)
	values := b.linearBias(x, attn.vProj, attn.vBias)

	queries = b.rmsNorm(queries, attn.qNorm, eps)
	queries = b.transpose(queries, []int{0, 2, 1, 3})
	keys = b.reshape(keys, []int{batch, L, nkv, hd})
	keys = b.rmsNorm(keys, attn.kNorm, eps)
	keys = b.transpose(keys, []int{0, 2, 1, 3})
	values = b.reshape(values, []int{batch, L, nkv, hd})
	values = b.transpose(values, []int{0, 2, 1, 3})

	if ropeOff == nil {
		queries = b.rope(queries, ropeDims, theta, offset)
		keys = b.rope(keys, ropeDims, theta, offset)
	} else {
		queries = b.ropePerRow(queries, ropeOff, func(r *mlxgo.Array, o int) *mlxgo.Array { return b.rope(r, ropeDims, theta, o) })
		keys = b.ropePerRow(keys, ropeOff, func(r *mlxgo.Array, o int) *mlxgo.Array { return b.rope(r, ropeDims, theta, o) })
	}
	if b.err == nil {
		keys, values, b.err = cache.Update(keys, values, b.s)
	}

	out := b.sdpaWith(queries, keys, values, scale, mode, mask)
	out = b.transpose(out, []int{0, 2, 1, 3})
	out = b.reshape(out, []int{batch, L, nh * hd})
	out = b.mul(out, b.sigmoidArr(gate))
	return b.linearBias(out, attn.oProj, attn.oBias)
}

// gatedDeltaNet is the recurrent token mixer. It projects the input into per-head
// query, key, value, and gate streams, runs a depthwise causal convolution over
// the q/k/v bands, then advances a per-head recurrent state one timestep at a
// time. This is the ops-based recurrence; the fused metal kernel the reference
// also offers is a later throughput optimization. The leading axis of every
// stream and of the carried state is the batch axis: a synchronized decode step
// (L==1) advances every row's recurrent state in parallel by broadcasting over it.
// For a left-padded ragged prefill the per-position SSM mask (pos >= leftPad[b],
// the reference's conv-cache make_mask) drops the front padding two ways: it zeros
// the padding positions of the convolution input so they do not leak into a real
// position's causal window, and it restores the prior state at each padding timestep
// so the recurrence skips it. The mask only matters on a prefill (L > 1); a decode
// step carries one real token per row, so a uniform or decoding cohort keeps the
// identity path.
func (b *fb) gatedDeltaNet(x *mlxgo.Array, layer *qwen3NextLinear, a *Qwen3NextArgs, cache *KVTensorCache, leftPad []int, batch, L int) *mlxgo.Array {
	nk := a.LinearNumKeyHeads
	nv := a.LinearNumValueHeads
	dk := a.LinearKeyHeadDim
	dv := a.LinearValueHeadDim
	keyDim := a.KeyDim()
	valueDim := a.ValueDim()
	convDim := a.ConvDim()
	kSize := a.LinearConvKernelDim
	vPerK := nv / nk

	// Fused input projections, then fix_query_key_value_ordering: carve the
	// per-key-head qkvz block into q, k, v, and the output gate z.
	qkvz := b.linear(x, layer.inProjQKVZ)
	qkvz = b.reshape(qkvz, []int{batch, L, nk, 2*dk + 2*vPerK*dv})
	qkvzParts := b.splitLast(qkvz, []int{dk, 2 * dk, 2*dk + vPerK*dv})
	q := qkvzParts[0]
	k := qkvzParts[1]
	v := b.reshape(qkvzParts[2], []int{batch, L, nv, dv})
	z := b.reshape(qkvzParts[3], []int{batch, L, nv, dv})

	ba := b.linear(x, layer.inProjBA)
	ba = b.reshape(ba, []int{batch, L, nk, 2 * vPerK})
	baParts := b.splitLast(ba, []int{vPerK})
	betaPre := b.reshape(baParts[0], []int{batch, L, nv})
	alpha := b.reshape(baParts[1], []int{batch, L, nv})

	// Depthwise causal convolution over the concatenated q/k/v bands. The conv
	// state carries the kernel-1 trailing timesteps across decode steps.
	mixedQKV := b.concat([]*mlxgo.Array{
		b.reshape(q, []int{batch, L, keyDim}),
		b.reshape(k, []int{batch, L, keyDim}),
		b.reshape(v, []int{batch, L, valueDim}),
	}, 2)

	// The SSM mask is built only for a left-padded prefill; nil keeps the identity
	// path. It is a [batch, L, 1] float buffer of 1 at real positions and 0 at the
	// front padding, so the convolution zeroing and the per-step state restore are
	// plain multiplies and need no boolean-where kernel.
	var ssmMask *mlxgo.Array
	if leftPad != nil && L > 1 && b.err == nil {
		ssmMask, b.err = ssmLeftPadMask(leftPad, L, b.s)
	}
	if ssmMask != nil {
		mixedQKV = b.mul(mixedQKV, ssmMask)
	}

	convState := cache.ConvState()
	if convState == nil {
		convState = b.zeros([]int{batch, kSize - 1, convDim})
	}
	convInput := b.concat([]*mlxgo.Array{convState, mixedQKV}, 1)
	newConvState := b.sliceAxis(convInput, L, kSize-1, 1)
	convOut := b.silu(b.depthwiseConv1d(convInput, layer.conv1d, convDim, kSize, L))

	convParts := b.splitLast(convOut, []int{keyDim, 2 * keyDim})
	q = b.reshape(convParts[0], []int{batch, L, nk, dk})
	k = b.reshape(convParts[1], []int{batch, L, nk, dk})
	v = b.reshape(convParts[2], []int{batch, L, nv, dv})

	// Per-head query/key normalization with the fixed scale the reference folds
	// into q and k before the recurrence.
	invScale := float32(1.0 / math.Sqrt(float64(dk)))
	q = b.scalarMul(b.rmsNorm(q, nil, gatedDeltaEps), invScale*invScale)
	k = b.scalarMul(b.rmsNorm(k, nil, gatedDeltaEps), invScale)
	if vPerK > 1 {
		q = b.repeat(q, vPerK, 2)
		k = b.repeat(k, vPerK, 2)
	}

	beta := b.sigmoidArr(betaPre)
	g := b.computeG(layer.aLog, alpha, layer.dtBias)

	state := cache.SSMState()
	if state == nil {
		state = b.zeros([]int{batch, nv, dv, dk})
	}
	ys := make([]*mlxgo.Array, 0, L)
	for t := range L {
		// Each takeAt drops the seq axis; the leading axis stays batch and the
		// remaining singleton axes are the one-timestep broadcast dims.
		qt := b.reshape(b.takeAt(q, t, 1), []int{batch, nv, 1, dk})
		kt := b.reshape(b.takeAt(k, t, 1), []int{batch, nv, 1, dk})
		vt := b.reshape(b.takeAt(v, t, 1), []int{batch, nv, dv})
		gt := b.reshape(b.takeAt(g, t, 1), []int{batch, nv, 1, 1})
		bt := b.reshape(b.takeAt(beta, t, 1), []int{batch, nv, 1})

		oldState := state
		state = b.mul(state, gt)
		kvMem := b.sumAxis(b.mul(state, kt), 3, false)
		delta := b.mul(b.sub(vt, kvMem), bt)
		state = b.add(state, b.mul(kt, b.reshape(delta, []int{batch, nv, dv, 1})))
		yt := b.sumAxis(b.mul(state, qt), 3, false)
		if ssmMask != nil {
			// Restore the prior state at a padding timestep (mask 0), so the
			// recurrence skips it: state = old + mask*(state-old).
			mt := b.reshape(b.takeAt(ssmMask, t, 1), []int{batch, 1, 1, 1})
			state = b.add(oldState, b.mul(mt, b.sub(state, oldState)))
		}
		ys = append(ys, b.reshape(yt, []int{batch, 1, nv, dv}))
	}
	out := b.concat(ys, 1)

	cache.SetState(newConvState, state, L)

	out = b.mul(b.silu(z), b.rmsNorm(out, layer.norm, gatedDeltaEps))
	out = b.reshape(out, []int{batch, L, valueDim})
	return b.linear(out, layer.outProj)
}

// ssmLeftPadMask builds the gated delta net's per-position validity mask for a
// left-padded prefill block, a [batch, L, 1] float buffer with 1 at a real position
// and 0 at the front padding. It is the reference conv-cache make_mask, pos >=
// leftPad[b] over pos in [0, L), cast to the multiplicative form the convolution
// zeroing and the per-step state restore use. The trailing singleton broadcasts over
// the convolution channels, and each timestep slices its column to a [batch, 1, 1, 1]
// gate. The array is host-built (not a kernel), so it materializes on the default
// stub too.
func ssmLeftPadMask(leftPad []int, L int, s *mlxgo.Stream) (*mlxgo.Array, error) {
	batch := len(leftPad)
	data := make([]float32, batch*L)
	for b, pad := range leftPad {
		for t := range L {
			if t >= pad {
				data[b*L+t] = 1
			}
		}
	}
	return mlxgo.NewFloat32(data, batch, L, 1)
}

// depthwiseConv1d is the per-channel causal convolution the gated delta net runs
// over its q/k/v bands. The weight is [conv_dim, kernel, 1], one filter per
// channel; out[t, c] is the windowed sum of input[t+j, c] * weight[c, j, 0]. The
// input already carries the kernel-1 left context, so the valid output length is
// L. It is expressed as a kernel-length accumulate over shifted slices, which is
// exactly a stride-one depthwise convolution.
func (b *fb) depthwiseConv1d(x, weight *mlxgo.Array, convDim, kSize, L int) *mlxgo.Array {
	var acc *mlxgo.Array
	for j := range kSize {
		seg := b.sliceAxis(x, j, L, 1)
		wj := b.reshape(b.takeAt(weight, j, 1), []int{1, 1, convDim})
		term := b.mul(seg, wj)
		if acc == nil {
			acc = term
		} else {
			acc = b.add(acc, term)
		}
	}
	return acc
}

// computeG is the gated delta net decay gate: exp(-exp(A_log) * softplus(alpha +
// dt_bias)). softplus is logaddexp(x, 0). A_log and dt_bias are per value head and
// broadcast over the sequence.
func (b *fb) computeG(aLog, alpha, dtBias *mlxgo.Array) *mlxgo.Array {
	sp := b.logaddexp(b.add(alpha, dtBias), b.scalar(0))
	prod := b.mul(b.exp(aLog), sp)
	return b.exp(b.scalarMul(prod, -1))
}

// qwen3NextMoE is the sparse mixture block: a softmax router picks the top-k
// experts, the stacked SwitchGLU runs them, their outputs are score-weighted and
// summed, and a sigmoid-gated shared expert is added.
func (b *fb) qwen3NextMoE(x *mlxgo.Array, mlp *qwen3NextMLP, a *Qwen3NextArgs, batch, L int) *mlxgo.Array {
	ne := a.NumExperts
	k := a.NumExpertsPerTok
	gates := b.softmax(b.linear(x, mlp.gate), -1)
	inds := b.sliceLastK(b.argpartition(gates, -k, -1), k, ne, 2)
	scores := b.takeAlongAxis(gates, inds, 2)
	if a.NormTopkProb {
		scores = b.div(scores, b.sumAxis(scores, 2, true))
	}
	y := b.switchGLU(x, mlp.switchGate, mlp.switchUp, mlp.switchDown, inds)
	y = b.sumAxis(b.mul(y, b.reshape(scores, []int{batch, L, k, 1})), 2, false)

	shared := b.swiglu(x, mlp.sharedGate, mlp.sharedUp, mlp.sharedDown)
	shared = b.mul(b.sigmoidArr(b.linear(x, mlp.sharedExpertGate)), shared)
	return b.add(y, shared)
}

// swiglu is the SwiGLU feed-forward down_proj(silu(gate_proj(x)) * up_proj(x)),
// the dense MLP and the shared expert both use.
func (b *fb) swiglu(x, gateW, upW, downW *mlxgo.Array) *mlxgo.Array {
	gate := b.silu(b.linear(x, gateW))
	up := b.linear(x, upW)
	return b.linear(b.mul(gate, up), downW)
}

// sliceLastK gathers the last k positions along axis (the top-k slots an
// argpartition leaves at the high end). total is the axis length.
func (b *fb) sliceLastK(x *mlxgo.Array, k, total, axis int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	idx := make([]int32, k)
	for i := range idx {
		idx[i] = int32(total - k + i)
	}
	ia, err := mlxgo.NewInt32(idx, k)
	if err != nil {
		b.err = err
		return nil
	}
	return b.take(x, ia, axis)
}

// sliceAxis gathers n consecutive positions starting at start along axis.
func (b *fb) sliceAxis(x *mlxgo.Array, start, n, axis int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	idx := make([]int32, n)
	for i := range idx {
		idx[i] = int32(start + i)
	}
	ia, err := mlxgo.NewInt32(idx, n)
	if err != nil {
		b.err = err
		return nil
	}
	return b.take(x, ia, axis)
}

func (b *fb) sub(x, y *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Sub(x, y, b.s)
	b.err = err
	return r
}

func (b *fb) exp(x *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Exp(x, b.s)
	b.err = err
	return r
}

func (b *fb) logaddexp(x, y *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Logaddexp(x, y, b.s)
	b.err = err
	return r
}

func (b *fb) repeat(x *mlxgo.Array, repeats, axis int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Repeat(x, repeats, axis, b.s)
	b.err = err
	return r
}

func (b *fb) softmax(x *mlxgo.Array, axis int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Softmax(x, axis, b.s)
	b.err = err
	return r
}

func (b *fb) concat(arrs []*mlxgo.Array, axis int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Concatenate(arrs, axis, b.s)
	b.err = err
	return r
}

// zeros allocates a host-side zero array (no kernel), used to seed an empty
// recurrent state or convolution window on the first step.
func (b *fb) zeros(shape []int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Zeros(mlxgo.Float32, shape...)
	if err != nil {
		b.err = err
		return nil
	}
	return r
}
