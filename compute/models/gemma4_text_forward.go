// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"fmt"
	"math"

	"github.com/tamnd/fastmlx/mlxgo"
)

// gemma4Layer holds one decoder block's weight tensors. The key and value
// projections and the k norm are nil on a KV-shared layer (it reads an earlier
// layer's cache instead of owning one); v_proj is nil on a K-eq-V layer (it
// reuses the keys as the values). The per-layer-input gating tensors are nil
// when that path is off (hidden_size_per_layer_input is zero).
type gemma4Layer struct {
	inputLayernorm          *mlxgo.Array
	postAttentionLayernorm  *mlxgo.Array
	preFeedforwardLayernorm *mlxgo.Array
	postFeedforwardLayernrm *mlxgo.Array
	qProj                   *qLinear
	kProj                   *qLinear
	vProj                   *qLinear
	oProj                   *qLinear
	qNorm                   *mlxgo.Array
	kNorm                   *mlxgo.Array
	gateProj                *qLinear
	upProj                  *qLinear
	downProj                *qLinear
	layerScalar             *mlxgo.Array
	perLayerInputGate       *qLinear
	perLayerProjection      *qLinear
	postPerLayerInputNorm   *mlxgo.Array
}

// Gemma4TextModel is an assembled Gemma 4 text decoder: the decoded args plus
// the weight tensors wired into typed fields. It is the most structurally varied
// decoder in the set; the per-layer choices (sliding vs full attention, KV
// sharing, dual head dim, K-eq-V, partial rotary) are all read off the args, and
// the constructor pulls only the tensors each layer actually owns.
type Gemma4TextModel struct {
	args        *Gemma4TextArgs
	embedTokens *qLinear
	norm        *mlxgo.Array
	lmHead      *qLinear // nil when the head is tied to the embedding table

	// Per-layer-input gating tensors, nil when that path is off.
	embedTokensPerLayer     *qLinear
	perLayerModelProjection *qLinear
	perLayerProjectionNorm  *mlxgo.Array

	layers []gemma4Layer

	// fullFreqs is the proportional-rotary frequency table the full-attention
	// layers share (nil when there are no full-attention layers). Sliding layers
	// use a plain single-base rope and need no table.
	fullFreqs *mlxgo.Array
}

// NewGemma4TextModel wires a sanitized weight map into a runnable model. Every
// key the args declare in WeightNames must be present. The mixture-of-experts
// variant (enable_moe_block) is rejected here: its expert MLP needs a grouped
// matmul that the binding does not expose yet, so a MoE checkpoint fails at
// construction rather than mid-generation.
func NewGemma4TextModel(args *Gemma4TextArgs, weights map[string]*mlxgo.Array) (*Gemma4TextModel, error) {
	if args.EnableMoEBlock {
		return nil, fmt.Errorf("gemma4_text: enable_moe_block needs a grouped expert matmul that is not yet bound")
	}
	if args.FirstKVShared() <= 0 {
		return nil, fmt.Errorf("gemma4_text: no KV-owning layers (num_kv_shared_layers %d covers all %d layers)",
			args.NumKVSharedLayers, args.NumLayers())
	}
	get := func(name string) (*mlxgo.Array, error) {
		w, ok := weights[name]
		if !ok || w == nil {
			return nil, fmt.Errorf("gemma4_text: missing weight %q", name)
		}
		return w, nil
	}
	getQ := func(name string) (*qLinear, error) {
		return loadQLinear(weights, name, args.quant)
	}

	m := &Gemma4TextModel{args: args, layers: make([]gemma4Layer, args.NumLayers())}
	var err error
	if m.embedTokens, err = getQ("model.embed_tokens"); err != nil {
		return nil, err
	}
	if m.norm, err = get("model.norm.weight"); err != nil {
		return nil, err
	}
	if args.HasPerLayerInputs() {
		if m.embedTokensPerLayer, err = getQ("model.embed_tokens_per_layer"); err != nil {
			return nil, err
		}
		if m.perLayerModelProjection, err = getQ("model.per_layer_model_projection"); err != nil {
			return nil, err
		}
		if m.perLayerProjectionNorm, err = get("model.per_layer_projection_norm.weight"); err != nil {
			return nil, err
		}
	}

	for i := range m.layers {
		p := fmt.Sprintf("model.layers.%d.", i)
		L := &m.layers[i]
		norms := []struct {
			name string
			dst  **mlxgo.Array
		}{
			{p + "input_layernorm.weight", &L.inputLayernorm},
			{p + "post_attention_layernorm.weight", &L.postAttentionLayernorm},
			{p + "pre_feedforward_layernorm.weight", &L.preFeedforwardLayernorm},
			{p + "post_feedforward_layernorm.weight", &L.postFeedforwardLayernrm},
			{p + "self_attn.q_norm.weight", &L.qNorm},
			{p + "layer_scalar", &L.layerScalar},
		}
		for _, f := range norms {
			if *f.dst, err = get(f.name); err != nil {
				return nil, err
			}
		}
		proj := []struct {
			name string
			dst  **qLinear
		}{
			{p + "self_attn.q_proj", &L.qProj},
			{p + "self_attn.o_proj", &L.oProj},
			{p + "mlp.gate_proj", &L.gateProj},
			{p + "mlp.up_proj", &L.upProj},
			{p + "mlp.down_proj", &L.downProj},
		}
		for _, f := range proj {
			if *f.dst, err = getQ(f.name); err != nil {
				return nil, err
			}
		}
		if args.HasKV(i) {
			if L.kProj, err = getQ(p + "self_attn.k_proj"); err != nil {
				return nil, err
			}
			if L.kNorm, err = get(p + "self_attn.k_norm.weight"); err != nil {
				return nil, err
			}
			if !args.UseKEqV(i) {
				if L.vProj, err = getQ(p + "self_attn.v_proj"); err != nil {
					return nil, err
				}
			}
		}
		if args.HasPerLayerInputs() {
			if L.perLayerInputGate, err = getQ(p + "per_layer_input_gate"); err != nil {
				return nil, err
			}
			if L.perLayerProjection, err = getQ(p + "per_layer_projection"); err != nil {
				return nil, err
			}
			if L.postPerLayerInputNorm, err = get(p + "post_per_layer_input_norm.weight"); err != nil {
				return nil, err
			}
		}
	}

	if args.TieWordEmbeddings {
		m.lmHead = nil
	} else if m.lmHead, err = getQ("lm_head"); err != nil {
		return nil, err
	}

	if fi := firstFullLayer(args); fi >= 0 {
		freqs := proportionalFreqs(args.PerLayerHeadDim(fi), args.LayerPartialRotary(fi), args.LayerRopeTheta(fi))
		if m.fullFreqs, err = mlxgo.NewFloat32(freqs, len(freqs)); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// firstFullLayer returns the index of the first full-attention layer, or -1 when
// every layer is sliding.
func firstFullLayer(a *Gemma4TextArgs) int {
	for i := range a.NumLayers() {
		if !a.IsSliding(i) {
			return i
		}
	}
	return -1
}

// proportionalFreqs builds the rotary frequency table for a full-attention layer:
// the partial-rotary scheme rotates only the leading rotated = int(head_dim *
// factor) dimensions, so the first rotated/2 frequencies are base raised to the
// even-index exponents (over the full head_dim) and the remaining tail is +Inf,
// which leaves those dimension pairs unrotated. This mirrors mlx_lm's
// ProportionalRoPE with its default factor of 1.
func proportionalFreqs(headDim int, partial, base float64) []float32 {
	half := headDim / 2
	rotated := int(float64(headDim) * partial)
	freqs := make([]float32, half)
	k := 0
	for j := 0; j < rotated && k < half; j += 2 {
		freqs[k] = float32(math.Pow(base, float64(j)/float64(headDim)))
		k++
	}
	for ; k < half; k++ {
		freqs[k] = float32(math.Inf(1))
	}
	return freqs
}

// slidingWindowMask builds the additive attention mask for a sliding-window layer
// over a full (non-rotating) key cache: query position p = offset+i may attend to
// key j only when j <= p (causal) and p-j < window (inside the window); every
// other entry is -Inf so softmax drops it. The shape is [1, 1, qLen, offset+qLen]
// so it broadcasts across batch and heads. A rotating key cache would store only
// the window and need no such mask; keeping the full cache and masking yields the
// identical attention and defers that memory optimization.
func slidingWindowMask(qLen, offset, window int) (*mlxgo.Array, error) {
	total := offset + qLen
	data := make([]float32, qLen*total)
	neg := float32(math.Inf(-1))
	for i := range qLen {
		p := offset + i
		row := i * total
		for j := range total {
			if j > p || p-j >= window {
				data[row+j] = neg
			}
		}
	}
	return mlxgo.NewFloat32(data, 1, 1, qLen, total)
}

// batchLeftPadSlidingWindowMask is the per-row sliding-window mask for a
// left-padded ragged cohort, shaped [batch, 1, qLen, offset+qLen]. It adds the
// left-padding term to slidingWindowMask: row b's query position p = offset+i may
// attend key j only when j is at or before p (causal), within the window
// (p-j < window), and at or after that row's left padding (j >= leftPad[b]), so
// the padding keys prepended to a short prompt stay masked here exactly as in the
// full-attention mask. A row with no padding reduces byte for byte to the
// single-sequence slidingWindowMask, so this is a strict generalization, and it
// uses the same negative infinity so the reduction is exact.
func batchLeftPadSlidingWindowMask(leftPad []int, qLen, offset, window int) (*mlxgo.Array, error) {
	total := offset + qLen
	data := make([]float32, len(leftPad)*qLen*total)
	neg := float32(math.Inf(-1))
	for b, pad := range leftPad {
		base := b * qLen * total
		for i := range qLen {
			p := offset + i
			row := base + i*total
			for j := range total {
				if j > p || p-j >= window || j < pad {
					data[row+j] = neg
				}
			}
		}
	}
	return mlxgo.NewFloat32(data, len(leftPad), 1, qLen, total)
}

// gemma4Intermediate carries one owning layer's post-rope, post-cache keys and
// values plus the rope offset, so a later KV-shared layer of the same kind can
// reuse them exactly as the reference threads previous_kvs.
type gemma4Intermediate struct {
	keys, values *mlxgo.Array
	offset       int
}

// Forward runs a single sequence's tokens through the model and returns the
// logits, shaped [1, len(tokens), vocab_size]. caches holds one KVTensorCache per
// layer; only the KV-owning layers (the first FirstKVShared) append to theirs,
// and each KV-shared layer reads the cache its PreviousKVs entry points at.
func (m *Gemma4TextModel) Forward(tokens []int32, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	return m.forwardBL(tokens, 1, len(tokens), caches, s)
}

// BatchDecode runs one decode step for batch sequences at once and returns the
// logits, shaped [batch, 1, vocab_size]. tokens holds the batch's single tokens
// in row order, the [batch, 1] decode input the reference forms with
// inputs[:, None]. Every sequence shares the same cache length (a synchronized
// batch), so with L == 1 the step needs no mask and the [1, 1, 1, total] sliding
// window mask broadcasts across the batch.
func (m *Gemma4TextModel) BatchDecode(tokens []int32, batch int, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	return m.forwardBL(tokens, batch, 1, caches, s)
}

// forwardBL is the batch-polymorphic forward shared by Forward and BatchDecode.
// tokens is the row-major [batch, L] token matrix flattened to batch*L values
// and the result is [batch, L, vocab_size]; batch == 1 reproduces the
// single-sequence shapes and L == 1 is the batched decode step.
func (m *Gemma4TextModel) forwardBL(tokens []int32, batch, L int, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
	if len(caches) != len(m.layers) {
		return nil, fmt.Errorf("gemma4_text: got %d caches, want %d", len(caches), len(m.layers))
	}
	a := m.args
	eps := float32(a.RMSNormEps)
	nh := a.NumAttentionHeads
	prev := a.PreviousKVs()

	b := &fb{s: s}

	ids, err := mlxgo.NewInt32(tokens, batch*L)
	if err != nil {
		return nil, err
	}

	// Token embedding, scaled by sqrt(hidden_size), with a leading batch axis.
	h := b.qembed(m.embedTokens, ids)
	h = b.reshape(h, []int{batch, L, a.HiddenSize})
	h = b.scalarMul(h, float32(a.EmbedScale()))

	// Per-layer inputs: the per-layer embedding table and a projection of the
	// scaled hidden state are normalized, averaged, and sliced per layer.
	var perLayerInputs *mlxgo.Array
	if a.HasPerLayerInputs() {
		nl := a.NumLayers()
		hp := a.HiddenSizePerLayerIn
		embedScale, gateScale, projScale := a.PerLayerInputScales()

		pin := b.qembed(m.embedTokensPerLayer, ids)
		pin = b.reshape(pin, []int{batch, L, nl, hp})
		pin = b.scalarMul(pin, float32(embedScale))

		proj := b.qlinear(h, m.perLayerModelProjection)
		proj = b.scalarMul(proj, float32(projScale))
		proj = b.reshape(proj, []int{batch, L, nl, hp})
		proj = b.rmsNorm(proj, m.perLayerProjectionNorm, eps)

		perLayerInputs = b.scalarMul(b.add(proj, pin), float32(gateScale))
	}

	// The sliding-window mask depends only on the step (query length and the
	// pre-step offset), so build it once and share it across sliding layers. The
	// owning offset is uniform because every owning cache grows by L each step.
	// For a left-padded ragged cohort both masks gain a per-row term: the full
	// layers read the left-padded causal mask from the cache, and the sliding
	// layers use the per-row sliding-window mask so prepended padding stays masked.
	stepOffset := caches[0].Offset
	leftPad := caches[0].LeftPad()
	ropeOff := caches[0].RopeOffsets()
	fullMode, fullMask, err := caches[0].AttnMask(batch, L, s)
	if err != nil {
		return nil, err
	}
	var slideMask *mlxgo.Array
	for i := range m.layers {
		if a.IsSliding(i) {
			if leftPad == nil {
				slideMask, err = slidingWindowMask(L, stepOffset, a.SlidingWindow)
			} else {
				slideMask, err = batchLeftPadSlidingWindowMask(leftPad, L, stepOffset, a.SlidingWindow)
			}
			if err != nil {
				return nil, err
			}
			break
		}
	}

	intermediates := make([]gemma4Intermediate, len(m.layers))
	for i := range m.layers {
		layer := &m.layers[i]
		hd := a.PerLayerHeadDim(i)
		nkv := a.PerLayerNumKVHeads(i)

		// Attention.
		x := b.rmsNorm(h, layer.inputLayernorm, eps)
		q := b.qlinear(x, layer.qProj)
		q = b.reshape(q, []int{batch, L, nh, hd})
		q = b.rmsNorm(q, layer.qNorm, eps)
		q = b.transpose(q, []int{0, 2, 1, 3})

		var keys, values *mlxgo.Array
		var offset int
		if a.HasKV(i) {
			offset = caches[i].Offset
			k := b.qlinear(x, layer.kProj)
			k = b.reshape(k, []int{batch, L, nkv, hd})
			k = b.rmsNorm(k, layer.kNorm, eps)
			k = b.transpose(k, []int{0, 2, 1, 3})
			if ropeOff == nil {
				k = m.ropeLayer(b, i, k, hd, offset)
			} else {
				k = b.ropePerRow(k, ropeOff, func(r *mlxgo.Array, o int) *mlxgo.Array { return m.ropeLayer(b, i, r, hd, o) })
			}

			vSrc := layer.vProj
			var v *mlxgo.Array
			if vSrc == nil { // K-eq-V: values are the keys, before the key norm and rope.
				v = b.qlinear(x, layer.kProj)
			} else {
				v = b.qlinear(x, vSrc)
			}
			v = b.reshape(v, []int{batch, L, nkv, hd})
			v = b.rmsNorm(v, nil, eps) // v_norm is scale-free (no weight).
			v = b.transpose(v, []int{0, 2, 1, 3})

			if b.err == nil {
				keys, values, b.err = caches[i].Update(k, v, s)
			}
			intermediates[i] = gemma4Intermediate{keys: keys, values: values, offset: offset}
		} else {
			shared := intermediates[prev[i]]
			keys, values, offset = shared.keys, shared.values, shared.offset
		}

		if ropeOff == nil {
			q = m.ropeLayer(b, i, q, hd, offset)
		} else {
			q = b.ropePerRow(q, ropeOff, func(r *mlxgo.Array, o int) *mlxgo.Array { return m.ropeLayer(b, i, r, hd, o) })
		}

		maskMode, mask := "", slideMask
		if !a.IsSliding(i) {
			maskMode, mask = fullMode, fullMask
		}
		attn := b.sdpaWith(q, keys, values, 1.0, maskMode, mask)
		attn = b.transpose(attn, []int{0, 2, 1, 3})
		attn = b.reshape(attn, []int{batch, L, nh * hd})
		attn = b.qlinear(attn, layer.oProj)
		attn = b.rmsNorm(attn, layer.postAttentionLayernorm, eps)
		h = b.add(h, attn)

		// Gated MLP, wrapped in the pre/post feedforward norms.
		residual := h
		y := b.rmsNorm(h, layer.preFeedforwardLayernorm, eps)
		y = b.qlinear(b.geglu(b.qlinear(y, layer.gateProj), b.qlinear(y, layer.upProj)), layer.downProj)
		y = b.rmsNorm(y, layer.postFeedforwardLayernrm, eps)
		h = b.add(residual, y)

		// Per-layer input gating.
		if a.HasPerLayerInputs() {
			pli := b.takeAt(perLayerInputs, i, 2)
			pli = b.reshape(pli, []int{batch, L, a.HiddenSizePerLayerIn})
			gate := b.geluApprox(b.qlinear(h, layer.perLayerInputGate))
			gate = b.mul(gate, pli)
			gate = b.qlinear(gate, layer.perLayerProjection)
			gate = b.rmsNorm(gate, layer.postPerLayerInputNorm, eps)
			h = b.add(h, gate)
		}

		h = b.mul(h, layer.layerScalar)
	}

	h = b.rmsNorm(h, m.norm, eps)
	head := m.lmHead
	if head == nil {
		head = m.embedTokens
	}
	logits := b.qlinear(h, head)
	if a.FinalLogitSoftcapping > 0 {
		logits = b.softcap(logits, float32(a.FinalLogitSoftcapping))
	}
	if b.err != nil {
		return nil, b.err
	}
	return logits, nil
}

// ropeLayer applies the layer's rotary embedding: a full-attention layer uses the
// shared proportional frequency table (partial rotation); a sliding layer uses a
// plain single-base rope over its whole head dim.
func (m *Gemma4TextModel) ropeLayer(b *fb, i int, x *mlxgo.Array, headDim, offset int) *mlxgo.Array {
	if m.args.IsSliding(i) {
		return b.rope(x, headDim, float32(m.args.LayerRopeTheta(i)), offset)
	}
	return b.ropeFreqs(x, headDim, offset, m.fullFreqs)
}

// The builder methods below extend fb with the ops the Gemma decoder needs beyond
// the dense-transformer set: scalar broadcasts, the tanh-approximation GELU and
// its gated form, the partial-rotary rope, an explicit-mask attention, and the
// final-logit soft cap.

// take gathers rows of x along axis using an index array.
func (b *fb) take(x, indices *mlxgo.Array, axis int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Take(x, indices, axis, b.s)
	b.err = err
	return r
}

// takeAt gathers the single slice at position index along axis, keeping the axis
// (length one) so the caller reshapes it away.
func (b *fb) takeAt(x *mlxgo.Array, index, axis int) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	idx, err := mlxgo.NewInt32([]int32{int32(index)}, 1)
	if err != nil {
		b.err = err
		return nil
	}
	return b.take(x, idx, axis)
}

// scalar builds a one-element float array for a broadcast operand.
func (b *fb) scalar(c float32) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.NewFloat32([]float32{c}, 1)
	if err != nil {
		b.err = err
		return nil
	}
	return r
}

func (b *fb) scalarMul(x *mlxgo.Array, c float32) *mlxgo.Array { return b.mul(x, b.scalar(c)) }
func (b *fb) scalarAdd(x *mlxgo.Array, c float32) *mlxgo.Array { return b.add(x, b.scalar(c)) }

func (b *fb) scalarDiv(x *mlxgo.Array, c float32) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Div(x, b.scalar(c), b.s)
	b.err = err
	return r
}

func (b *fb) tanh(x *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.Tanh(x, b.s)
	b.err = err
	return r
}

// geluApprox is the tanh approximation of GELU,
// 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), matching mlx_lm's gelu_approx.
func (b *fb) geluApprox(x *mlxgo.Array) *mlxgo.Array {
	x3 := b.mul(b.mul(x, x), x)
	inner := b.add(x, b.scalarMul(x3, 0.044715))
	t := b.tanh(b.scalarMul(inner, 0.7978845608028654))
	return b.scalarMul(b.mul(x, b.scalarAdd(t, 1)), 0.5)
}

// geglu is the GELU-gated linear unit, gelu_approx(gate) * x.
func (b *fb) geglu(gate, x *mlxgo.Array) *mlxgo.Array { return b.mul(b.geluApprox(gate), x) }

// ropeFreqs applies rotary embedding from an explicit frequency table (the
// partial-rotary path for full-attention layers).
func (b *fb) ropeFreqs(x *mlxgo.Array, dims, offset int, freqs *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.RoPEWithFreqs(x, dims, false, 1, offset, freqs, b.s)
	b.err = err
	return r
}

// sdpaWith is attention with an explicit additive mask (used for the sliding
// window); a nil mask falls back to the built-in maskMode.
func (b *fb) sdpaWith(q, k, v *mlxgo.Array, scale float32, maskMode string, mask *mlxgo.Array) *mlxgo.Array {
	if b.err != nil {
		return nil
	}
	r, err := mlxgo.ScaledDotProductAttention(q, k, v, scale, maskMode, mask, b.s)
	b.err = err
	return r
}

// softcap is the final-logit soft cap, tanh(x/cap) * cap.
func (b *fb) softcap(x *mlxgo.Array, cap float32) *mlxgo.Array {
	return b.scalarMul(b.tanh(b.scalarDiv(x, cap)), cap)
}
