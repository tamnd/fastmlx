// SPDX-License-Identifier: MIT OR Apache-2.0

// Package quant ports the pure decision logic of the universal dynamic
// quantization scheme (oQ): the per-tensor quantize-or-keep predicate, the
// level/bits/bpw tables, and the serialized-size and effective-bits-per-weight
// estimators. These are functions of a tensor's module path, its shape, and the
// model config alone, so they run without the MLX backend. The actual quantize
// op (and the calibration-driven sensitivity plan it feeds) runs on the step
// thread once the compute layer lands.
package quant

import (
	"maps"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// OQLevels is the set of valid oQ levels.
var OQLevels = map[float64]bool{2: true, 3: true, 3.5: true, 4: true, 5: true, 6: true, 8: true}

// DefaultGroupSize is the default quantization group size.
const DefaultGroupSize = 64

// levelBits maps an oQ level to its base bit width (3.5 shares 3-bit base).
var levelBits = map[float64]int{2: 2, 3: 3, 3.5: 3, 4: 4, 5: 5, 6: 6, 8: 8}

// levelProtection records the protection profile per level; every supported
// level uses "full" protection.
var levelProtection = map[float64]string{2: "full", 3: "full", 3.5: "full", 4: "full", 5: "full", 6: "full", 8: "full"}

// bpwTargets maps an oQ level to its (target, hard-cap) bits-per-weight.
var bpwTargets = map[float64][2]float64{
	2:   {2.8, 3.0},
	3:   {3.5, 3.7},
	3.5: {3.8, 4.0},
	4:   {4.6, 4.7},
	5:   {5.5, 5.7},
	6:   {6.5, 6.7},
}

// BPWTargetsForLevel returns the (target, hard-cap) bits-per-weight for a level
// and whether the level has a target.
func BPWTargetsForLevel(oqLevel float64) (target, hardCap float64, ok bool) {
	t, present := bpwTargets[oqLevel]
	if !present {
		return 0, 0, false
	}
	return t[0], t[1], true
}

// MandatoryBoostPatterns are the consensus-critical tensors that the
// byte-budgeted planner pre-allocates to 8-bit before distributing the rest.
var MandatoryBoostPatterns = map[string]map[string]any{
	"lm_head":      {"bits": 8, "group_size": 64, "mode": "affine"},
	"embeddings":   {"bits": 8, "group_size": 64, "mode": "affine"},
	"embed_tokens": {"bits": 8, "group_size": 64, "mode": "affine"},
	"wte":          {"bits": 8, "group_size": 64, "mode": "affine"},
}

var skipQuantPatterns = []string{"layernorm", "rmsnorm", "norm.weight", "norm.bias", "ln_", "layer_norm"}

// Kind distinguishes the three predicate outcomes.
type Kind int

const (
	// KeepFP16 means skip quantization, leave the tensor full precision.
	KeepFP16 Kind = iota
	// UseDefault means quantize at the level's default bits.
	UseDefault
	// UseOverride means quantize with the per-tensor Override settings.
	UseOverride
)

// Result is the per-tensor decision: KeepFP16 (the reference's False),
// UseDefault (True), or UseOverride carrying the bits/group_size/mode dict.
type Result struct {
	Kind     Kind
	Override map[string]any
}

func skip() Result       { return Result{Kind: KeepFP16} }
func useDefault() Result { return Result{Kind: UseDefault} }
func override(m map[string]any) Result {
	return Result{Kind: UseOverride, Override: m}
}

// Config is a model config.json dict, as decoded from JSON. Nested objects are
// map[string]any and numbers are float64, matching encoding/json.
type Config map[string]any

func asInt(v any) (int, bool) {
	switch x := v.(type) {
	case float64:
		return int(x), true
	case int:
		return x, true
	case int64:
		return int(x), true
	}
	return 0, false
}

func (c Config) textConfig() Config {
	if tc, ok := c["text_config"].(map[string]any); ok {
		return Config(tc)
	}
	return Config{}
}

// numLayers mirrors `config.get("num_hidden_layers") or tc.get(..., 32)`.
func numLayers(c, tc Config) int {
	if i, ok := asInt(c["num_hidden_layers"]); ok && i != 0 {
		return i
	}
	if v, present := tc["num_hidden_layers"]; present {
		if i, ok := asInt(v); ok {
			return i
		}
	}
	return 32
}

// numExperts mirrors the four-way `or` chain over the expert-count keys.
func numExperts(c, tc Config) int {
	for _, v := range []any{c["num_local_experts"], tc["num_local_experts"], c["num_experts"], tc["num_experts"]} {
		if i, ok := asInt(v); ok && i != 0 {
			return i
		}
	}
	return 0
}

// hiddenSize mirrors `config.get("hidden_size") or tc.get("hidden_size", 0)`.
func hiddenSize(c, tc Config) int {
	if i, ok := asInt(c["hidden_size"]); ok && i != 0 {
		return i
	}
	if i, ok := asInt(tc["hidden_size"]); ok {
		return i
	}
	return 0
}

func (c Config) stringSet(key string) map[string]bool {
	set := map[string]bool{}
	switch v := c[key].(type) {
	case []any:
		for _, e := range v {
			if s, ok := e.(string); ok {
				set[s] = true
			}
		}
	case []string:
		for _, s := range v {
			set[s] = true
		}
	case map[string]any:
		for k := range v {
			set[k] = true
		}
	}
	return set
}

func (c Config) truthy(key string) bool {
	switch v := c[key].(type) {
	case nil:
		return false
	case bool:
		return v
	case float64:
		return v != 0
	case int:
		return v != 0
	case string:
		return v != ""
	case []any:
		return len(v) > 0
	case map[string]any:
		return len(v) > 0
	default:
		return true
	}
}

var layerIndexRe = regexp.MustCompile(`layers\.(\d+)\.`)

// ExtractLayerIndex returns the transformer layer index in a module path, or -1.
func ExtractLayerIndex(path string) int {
	m := layerIndexRe.FindStringSubmatch(path)
	if m == nil {
		return -1
	}
	n, _ := strconv.Atoi(m[1])
	return n
}

// NormalizeQuantPath strips a trailing .weight/.scales/.biases suffix so the
// path matches the module path used in configs.
func NormalizeQuantPath(path string) string {
	for _, suf := range []string{".weight", ".scales", ".biases"} {
		if strings.HasSuffix(path, suf) {
			return path[:len(path)-len(suf)]
		}
	}
	return path
}

// IsVisionTensor reports whether a tensor belongs to the vision encoder/projector.
func IsVisionTensor(name string) bool {
	for _, p := range []string{"visual.", "vision_", "patch_embed", "pos_embed", "image_newline", "multi_modal_projector", "visual.merger", "image_norm", "temporal_embed"} {
		if strings.Contains(name, p) {
			return true
		}
	}
	return false
}

// IsAudioTensor reports whether a tensor belongs to the audio encoder. It
// matches audio_tower only, not embed_audio (which is quantized).
func IsAudioTensor(name string) bool {
	return strings.Contains(name, "audio_tower")
}

// IsMoERouter detects MoE router/gate layers, distinct from gate_proj.
func IsMoERouter(path string) bool {
	if strings.HasSuffix(path, "mlp.gate") || strings.HasSuffix(path, ".router") || strings.HasSuffix(path, ".router.layer") {
		return true
	}
	if strings.HasSuffix(path, ".gate") && !strings.Contains(path, "gate_proj") {
		return true
	}
	if strings.Contains(path, ".gate.") && !strings.Contains(path, "gate_proj") {
		return true
	}
	return false
}

// IsRoutedExpert reports whether a tensor belongs to routed MoE experts.
func IsRoutedExpert(path string) bool {
	if strings.Contains(path, "switch_mlp") {
		return true
	}
	if strings.Contains(path, "experts") && !strings.Contains(path, "shared_expert") {
		return true
	}
	if strings.Contains(path, "block_sparse_moe") && !strings.Contains(path, "shared_expert") {
		return true
	}
	return false
}

// DefaultBits reads the default quantization bits from config (4 if absent).
func DefaultBits(config Config) int {
	if q, ok := config["quantization"].(map[string]any); ok {
		if b, ok := asInt(q["bits"]); ok {
			return b
		}
	}
	return 4
}

// BaseBitsForLevel returns the base bit width for an oQ level.
func BaseBitsForLevel(oqLevel float64) int {
	if b, ok := levelBits[oqLevel]; ok {
		return b
	}
	return int(oqLevel)
}

// ModeForBits selects the quantization mode; always affine to minimize kernel
// combinations.
func ModeForBits(bits int) string { return "affine" }

// GSForMode returns the group size; always the default to minimize kernels.
func GSForMode(bits, defaultGS int) int { return defaultGS }

func bytesPerGroup(mode string) int {
	switch mode {
	case "mxfp4":
		return 1
	case "mxfp8":
		return 2
	default:
		return 4
	}
}

// TensorQuantizedBytes estimates the serialized bytes of a quantized tensor.
func TensorQuantizedBytes(shape []int, bits, groupSize int, mode string) int {
	nElements := 1
	for _, d := range shape {
		nElements *= d
	}
	if len(shape) < 2 {
		return nElements * 2
	}
	last := shape[len(shape)-1]
	if last%groupSize != 0 {
		return nElements * 2
	}
	rows := nElements / max(last, 1)
	nGroups := last / groupSize
	weightBytes := (nElements*bits + 7) / 8
	overheadBytes := rows * nGroups * bytesPerGroup(mode)
	return weightBytes + overheadBytes
}

// EstimateEffectiveBPW estimates the effective bits-per-weight over the
// quantizable weights, applying any per-path overrides.
func EstimateEffectiveBPW(namedShapes map[string][]int, baseBits, baseGroupSize int, baseMode string, overrides map[string]map[string]any) float64 {
	totalBits := 0
	totalParams := 0
	for path, shape := range namedShapes {
		nElements := 1
		for _, d := range shape {
			nElements *= d
		}
		totalParams += nElements

		bits, gs, mode := baseBits, baseGroupSize, baseMode
		if ov, ok := overrides[path]; ok {
			if b, ok := asInt(ov["bits"]); ok {
				bits = b
			}
			if g, ok := asInt(ov["group_size"]); ok {
				gs = g
			}
			if m, ok := ov["mode"].(string); ok {
				mode = m
			} else {
				mode = ModeForBits(bits)
			}
		}
		totalBits += 8 * TensorQuantizedBytes(shape, bits, gs, mode)
	}
	return float64(totalBits) / float64(max(totalParams, 1))
}

// SensitivityTier maps a sensitivity score to a boost tier: 4 (top), 2 (high),
// 1 (moderate).
func SensitivityTier(layerScore, maxScore float64) int {
	if maxScore <= 0 {
		return 1
	}
	ratio := layerScore / maxScore
	if ratio >= 0.5 {
		return 4
	}
	if ratio >= 0.2 {
		return 2
	}
	return 1
}

// EstimateMemory gives a rough peak-memory estimate for a streaming
// quantization run: the source mmap plus a fixed 6 GB headroom for the output
// buffer and sanitize overhead. The per-tensor estimate endpoint refines this.
func EstimateMemory(sourceSizeBytes int) map[string]any {
	peak := sourceSizeBytes + 6*1024*1024*1024
	return map[string]any{
		"peak_bytes":     peak,
		"peak_formatted": formatSize(peak),
	}
}

// formatSize renders a byte count as a human-readable string with a raw-bytes
// tier below 1 KB, then KB/MB/GB by magnitude with one decimal.
func formatSize(sizeBytes int) string {
	switch {
	case sizeBytes < 1024:
		return strconv.Itoa(sizeBytes) + " B"
	case sizeBytes < 1024*1024:
		return strconv.FormatFloat(float64(sizeBytes)/1024, 'f', 1, 64) + " KB"
	case sizeBytes < 1024*1024*1024:
		return strconv.FormatFloat(float64(sizeBytes)/(1024*1024), 'f', 1, 64) + " MB"
	default:
		return strconv.FormatFloat(float64(sizeBytes)/(1024*1024*1024), 'f', 1, 64) + " GB"
	}
}

// ValidateQuantizable reports whether a config indicates a quantizable model.
// Already-quantized models are excluded, except native FP8 which is full
// precision stored in FP8.
func ValidateQuantizable(config Config) bool {
	if _, ok := config["quantization"]; ok {
		return false
	}
	if qcRaw, ok := config["quantization_config"]; ok {
		if qc, ok := qcRaw.(map[string]any); ok {
			if qc["quant_method"] == "fp8" {
				return true
			}
		}
		return false
	}
	return true
}

// ShouldSkipTensor reports whether a tensor should be excluded from output. MTP
// tensors are stripped unless preserveMTP is set.
func ShouldSkipTensor(name string, preserveMTP bool) bool {
	if strings.Contains(name, ".mtp.") || strings.HasPrefix(name, "mtp.") {
		return !preserveMTP
	}
	return false
}

// IsMTPTensor reports whether a tensor key belongs to an MTP head.
func IsMTPTensor(name string) bool {
	return strings.HasPrefix(name, "mtp.") || strings.Contains(name, ".mtp.")
}

// IsMTPProtectedTensor reports whether a tensor inside the MTP head must stay
// full precision (aggressive quantization of these collapses draft acceptance).
func IsMTPProtectedTensor(name string) bool {
	if !(strings.HasPrefix(name, "mtp.") || strings.Contains(name, ".mtp.")) {
		return false
	}
	if strings.HasSuffix(name, "mtp.fc.weight") || strings.Contains(name, ".mtp.fc.weight") {
		return true
	}
	if strings.HasSuffix(name, ".e_proj.weight") || strings.HasSuffix(name, ".h_proj.weight") {
		return true
	}
	if strings.Contains(name, ".hc_head.") {
		return true
	}
	if strings.HasSuffix(name, ".hc_head_fn") || strings.HasSuffix(name, ".hc_head_base") || strings.HasSuffix(name, ".hc_head_scale") {
		return true
	}
	return false
}

// ShouldQuantizeTensor reports whether a tensor should be quantized based on its
// name and shape (norms and biases are skipped).
func ShouldQuantizeTensor(name string, shape []int) bool {
	if len(shape) < 2 {
		return false
	}
	nameLower := strings.ToLower(name)
	for _, p := range skipQuantPatterns {
		if strings.Contains(nameLower, p) {
			return false
		}
	}
	if strings.HasSuffix(name, ".bias") {
		return false
	}
	return true
}

// NormalizeMTPInConfig zeroes the MTP layer counts in the config in place, so a
// stripped-MTP output config presents itself as MTP-free.
func NormalizeMTPInConfig(config Config) {
	zero := func(m map[string]any) {
		for _, key := range []string{"mtp_num_hidden_layers", "num_nextn_predict_layers"} {
			if v, ok := m[key]; ok {
				if i, ok := asInt(v); ok && i != 0 {
					m[key] = 0
				}
			}
		}
	}
	zero(config)
	if tc, ok := config["text_config"].(map[string]any); ok {
		zero(tc)
	}
}

// GetPredicateBits returns the (bits, group_size, mode) for a tensor, and
// whether it is quantized at all. MTP-protected tensors and predicate skips
// return quantized=false.
func GetPredicateBits(tensorName string, config Config, oqLevel float64, groupSize int) (bits, gs int, mode string, quantized bool) {
	if IsMTPProtectedTensor(tensorName) {
		return 0, 0, "", false
	}
	baseBits := BaseBitsForLevel(oqLevel)
	res := UniversalQuantPredicate(tensorName, config, oqLevel)
	switch res.Kind {
	case KeepFP16:
		return 0, 0, "", false
	case UseOverride:
		b := baseBits
		if v, ok := asInt(res.Override["bits"]); ok {
			b = v
		}
		g := groupSize
		if v, ok := asInt(res.Override["group_size"]); ok {
			g = v
		}
		m := ModeForBits(b)
		if v, ok := res.Override["mode"].(string); ok {
			m = v
		}
		return b, g, m, true
	default:
		return baseBits, GSForMode(baseBits, groupSize), ModeForBits(baseBits), true
	}
}

// UniversalQuantPredicate is the per-tensor quantization decision based on the
// GGUF/unsloth/llama.cpp rules, parameterized by oQ level. It depends only on
// the module path and the config (the reference's module argument is unused).
func UniversalQuantPredicate(path string, config Config, oqLevel float64) Result {
	path = NormalizeQuantPath(path)
	pathL := strings.ToLower(path)

	if config.stringSet("_oq_non_quantizable")[path] {
		return skip()
	}

	tc := config.textConfig()
	numLayersV := numLayers(config, tc)
	numExpertsV := numExperts(config, tc)
	hiddenSizeV := hiddenSize(config, tc)
	isMoE := numExpertsV > 0

	baseBits := BaseBitsForLevel(oqLevel)
	fullProtection := levelProtection[oqLevel] == "full"
	if _, known := levelProtection[oqLevel]; !known {
		fullProtection = true // default "full"
	}

	gs := func() int {
		if IsMoERouter(path) {
			return 64
		}
		if numExpertsV >= 150 {
			return 128
		}
		return 64
	}
	bits := func(n int) Result {
		eff := max(n, baseBits)
		return override(map[string]any{"bits": eff, "group_size": GSForMode(eff, gs()), "mode": ModeForBits(eff)})
	}

	if IsMoERouter(path) {
		return skip()
	}
	if strings.Contains(path, "shared_expert_gate") && !strings.Contains(path, "gate_proj") {
		return override(map[string]any{"bits": 8, "group_size": 64, "mode": "affine"})
	}
	if IsVisionTensor(path) {
		return skip()
	}
	if IsAudioTensor(path) {
		return skip()
	}
	for _, p := range []string{"ssm_alpha", "ssm_beta", "a_log", "time_decay", "time_faaaa"} {
		if strings.Contains(pathL, p) {
			return skip()
		}
	}
	if strings.HasSuffix(path, ".D") {
		return skip()
	}
	if strings.HasSuffix(pathL, "dt_bias") {
		return skip()
	}
	if strings.Contains(pathL, "conv1d") && strings.Contains(pathL, "linear_attn") {
		return bits(8)
	}
	if strings.Contains(pathL, "linear_attn.out_proj") {
		return bits(5)
	}

	if boostMap, ok := config["_oq_boost_map"].(map[string]any); ok {
		if entry, ok := boostMap[path].(map[string]any); ok {
			cp := make(map[string]any, len(entry))
			maps.Copy(cp, entry)
			return override(cp)
		}
	}

	if config.truthy("_oq_use_budget_plan") {
		if strings.Contains(path, "ssm_output") || strings.Contains(path, "ssm_out") {
			return bits(8)
		}
		if strings.Contains(path, "lora.2") {
			return bits(8)
		}
		return useDefault()
	}

	if !fullProtection {
		if containsAny(path, "lm_head", "output.weight", "classifier") {
			return bits(6)
		}
		if containsAny(path, "ssm_output", "ssm_out") {
			return bits(8)
		}
		if containsAny(path, "embed_tokens", "wte", "word_embeddings") {
			return bits(baseBits + 2)
		}
		if numExpertsV >= 512 && hiddenSizeV >= 4096 {
			if strings.Contains(path, "gate_proj") && !strings.Contains(path, "shared_expert") {
				return bits(4)
			}
		}
		layerIdx := ExtractLayerIndex(path)
		if layerIdx >= 0 {
			sensitive := layerIdx < numLayersV/8 || layerIdx >= 7*numLayersV/8
			isExpert := strings.Contains(path, "switch_mlp") || strings.Contains(path, "experts")
			if sensitive && !isExpert {
				return bits(baseBits + 1)
			}
		}
		return useDefault()
	}

	if containsAny(path, "ssm_output", "ssm_out") {
		return bits(8)
	}
	if strings.Contains(path, "lora.2") {
		return bits(8)
	}
	if containsAny(path, "lm_head", "output.weight", "classifier") {
		return bits(6)
	}
	if strings.Contains(path, "cross_attn") && strings.Contains(path, "o_proj") {
		return bits(6)
	}
	if containsAny(path, "kv_a_proj_with_mqa", "kv_b_proj", "q_a_proj", "q_b_proj") {
		return bits(6)
	}
	if strings.Contains(path, "o_proj") && !strings.Contains(path, "shared_expert") {
		if !isMoE {
			return bits(5)
		}
	}
	if strings.Contains(path, "shared_expert") && !strings.HasSuffix(path, "shared_expert_gate") {
		return bits(8)
	}
	if numExpertsV >= 512 && hiddenSizeV >= 4096 {
		if strings.Contains(path, "gate_proj") && !strings.Contains(path, "shared_expert") {
			return bits(4)
		}
		if strings.Contains(path, "down_proj") && !strings.Contains(path, "shared_expert") {
			return bits(3)
		}
	}

	layerIdx := ExtractLayerIndex(path)

	var sensitive bool
	if smRaw, ok := config["_oq_sensitivity_map"].(map[string]any); ok && len(smRaw) > 0 && layerIdx >= 0 {
		scores := make([]float64, 0, len(smRaw))
		for _, v := range smRaw {
			if f, ok := toFloat(v); ok {
				scores = append(scores, f)
			}
		}
		sort.Sort(sort.Reverse(sort.Float64Slice(scores)))
		threshold := 0.0
		if len(scores) > 0 {
			idx := max(len(scores)/4-1, 0)
			threshold = scores[idx]
		}
		cur := 0.0
		if f, ok := toFloat(smRaw[strconv.Itoa(layerIdx)]); ok {
			cur = f
		}
		sensitive = cur >= threshold
	} else {
		sensitive = layerIdx >= 0 && (layerIdx < numLayersV/8 || layerIdx >= 7*numLayersV/8)
	}

	if containsAny(path, "v_proj", "v_a_proj", "v_b_proj") {
		if sensitive {
			return bits(6)
		}
		return useDefault()
	}
	if containsAny(path, "down_proj", "w2", "mlp.fc2", "wo") {
		isRoutedExpert := isMoE && !strings.Contains(path, "shared_expert") && (strings.Contains(path, "switch_mlp") || strings.Contains(path, "experts"))
		if isRoutedExpert {
			if oqLevel == 3.5 {
				return bits(4)
			}
			return useDefault()
		}
		if sensitive {
			return bits(6)
		}
		return bits(5)
	}
	if containsAny(path, "q_proj", "k_proj") {
		if sensitive {
			return bits(5)
		}
	}
	if containsAny(path, "qkv_proj", "in_proj_qkv", "attn_qkv") {
		if sensitive {
			return bits(5)
		}
	}
	if containsAny(path, "in_proj_z", "in_proj_a", "in_proj_b", "delta_net") {
		return bits(5)
	}
	if containsAny(path, "mixer.in_proj", "mixer.out_proj", "x_proj", "dt_proj") {
		return bits(5)
	}

	return useDefault()
}

// Snapshot renders the predicate result the way the reference predicate would
// serialize: false for KeepFP16, true for UseDefault, or the override object.
func (r Result) Snapshot() string {
	switch r.Kind {
	case KeepFP16:
		return "false"
	case UseDefault:
		return "true"
	default:
		return encodeOverride(r.Override)
	}
}

func containsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

func toFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int:
		return float64(x), true
	case int64:
		return float64(x), true
	}
	return 0, false
}

// encodeOverride serializes an override dict as a JSON object with bits,
// group_size, and mode in that order (the keys the predicate emits), so the
// snapshot is stable for comparison.
func encodeOverride(m map[string]any) string {
	var b strings.Builder
	b.WriteByte('{')
	first := true
	write := func(k string) {
		v, ok := m[k]
		if !ok {
			return
		}
		if !first {
			b.WriteByte(',')
		}
		first = false
		b.WriteString(strconv.Quote(k))
		b.WriteByte(':')
		switch x := v.(type) {
		case int:
			b.WriteString(strconv.Itoa(x))
		case float64:
			b.WriteString(strconv.FormatFloat(x, 'g', -1, 64))
		case string:
			b.WriteString(strconv.Quote(x))
		default:
			b.WriteString("null")
		}
	}
	write("bits")
	write("group_size")
	write("mode")
	b.WriteByte('}')
	return b.String()
}
