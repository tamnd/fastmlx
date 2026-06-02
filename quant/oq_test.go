// SPDX-License-Identifier: MIT OR Apache-2.0

package quant

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type oqFixture struct {
	Predicate []struct {
		Path   string          `json:"path"`
		Config map[string]any  `json:"config"`
		Level  float64         `json:"level"`
		Result json.RawMessage `json:"result"`
	} `json:"predicate"`
	LayerIndex      map[string]int             `json:"layer_index"`
	NormalizePath   map[string]string          `json:"normalize_path"`
	BaseBits        map[string]int             `json:"base_bits"`
	BPWTargets      map[string]json.RawMessage `json:"bpw_targets"`
	Validate        map[string]bool            `json:"validate"`
	SensitivityTier []struct {
		Score float64 `json:"score"`
		Maxs  float64 `json:"maxs"`
		Tier  int     `json:"tier"`
	} `json:"sensitivity_tier"`
	TensorBytes []struct {
		Shape []int  `json:"shape"`
		Bits  int    `json:"bits"`
		GS    int    `json:"gs"`
		Mode  string `json:"mode"`
		Bytes int    `json:"bytes"`
	} `json:"tensor_bytes"`
	EffectiveBPW   map[string]float64 `json:"effective_bpw"`
	ShouldQuantize []struct {
		Name  string `json:"name"`
		Shape []int  `json:"shape"`
		Q     bool   `json:"q"`
	} `json:"should_quantize"`
	ShouldSkip []struct {
		Name     string `json:"name"`
		Preserve bool   `json:"preserve"`
		Skip     bool   `json:"skip"`
	} `json:"should_skip"`
	MTPProtected  map[string]bool `json:"mtp_protected"`
	PredicateBits []struct {
		Name   string            `json:"name"`
		Config map[string]any    `json:"config"`
		Level  float64           `json:"level"`
		GS     int               `json:"gs"`
		Result []json.RawMessage `json:"result"`
	} `json:"predicate_bits"`
}

func loadOQFixture(t *testing.T) oqFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/oq.json")
	if err != nil {
		t.Fatal(err)
	}
	var f oqFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestUniversalQuantPredicateParity(t *testing.T) {
	fx := loadOQFixture(t)
	for _, c := range fx.Predicate {
		got := UniversalQuantPredicate(c.Path, Config(c.Config), c.Level).Snapshot()
		var g, w any
		if err := json.Unmarshal([]byte(got), &g); err != nil {
			t.Fatalf("predicate(%q) produced invalid JSON %q: %v", c.Path, got, err)
		}
		_ = json.Unmarshal(c.Result, &w)
		if !reflect.DeepEqual(g, w) {
			t.Errorf("predicate(%q, level=%v):\n got  %s\n want %s", c.Path, c.Level, got, string(c.Result))
		}
	}
}

func TestExtractLayerIndexParity(t *testing.T) {
	fx := loadOQFixture(t)
	for path, want := range fx.LayerIndex {
		if got := ExtractLayerIndex(path); got != want {
			t.Errorf("ExtractLayerIndex(%q) = %d, want %d", path, got, want)
		}
	}
}

func TestNormalizeQuantPathParity(t *testing.T) {
	fx := loadOQFixture(t)
	for path, want := range fx.NormalizePath {
		if got := NormalizeQuantPath(path); got != want {
			t.Errorf("NormalizeQuantPath(%q) = %q, want %q", path, got, want)
		}
	}
}

func TestBaseBitsForLevelParity(t *testing.T) {
	fx := loadOQFixture(t)
	for lvStr, want := range fx.BaseBits {
		lv := parseLevel(t, lvStr)
		if got := BaseBitsForLevel(lv); got != want {
			t.Errorf("BaseBitsForLevel(%v) = %d, want %d", lv, got, want)
		}
	}
}

func TestBPWTargetsForLevelParity(t *testing.T) {
	fx := loadOQFixture(t)
	for lvStr, raw := range fx.BPWTargets {
		lv := parseLevel(t, lvStr)
		target, hardCap, ok := BPWTargetsForLevel(lv)
		if string(raw) == "null" {
			if ok {
				t.Errorf("level %v: expected no target, got (%v,%v)", lv, target, hardCap)
			}
			continue
		}
		var want []float64
		if err := json.Unmarshal(raw, &want); err != nil {
			t.Fatal(err)
		}
		if !ok || target != want[0] || hardCap != want[1] {
			t.Errorf("level %v: got (%v,%v) ok=%v, want (%v,%v)", lv, target, hardCap, ok, want[0], want[1])
		}
	}
}

func TestValidateQuantizableParity(t *testing.T) {
	fx := loadOQFixture(t)
	configs := map[string]Config{
		"plain":         {"num_hidden_layers": 32.0},
		"already_quant": {"quantization": map[string]any{"bits": 4.0}},
		"qc_fp8":        {"quantization_config": map[string]any{"quant_method": "fp8"}},
		"qc_other":      {"quantization_config": map[string]any{"quant_method": "gptq"}},
	}
	for key, want := range fx.Validate {
		if got := ValidateQuantizable(configs[key]); got != want {
			t.Errorf("ValidateQuantizable(%s) = %v, want %v", key, got, want)
		}
	}
}

func TestSensitivityTierParity(t *testing.T) {
	fx := loadOQFixture(t)
	for _, c := range fx.SensitivityTier {
		if got := SensitivityTier(c.Score, c.Maxs); got != c.Tier {
			t.Errorf("SensitivityTier(%v,%v) = %d, want %d", c.Score, c.Maxs, got, c.Tier)
		}
	}
}

func TestTensorQuantizedBytesParity(t *testing.T) {
	fx := loadOQFixture(t)
	for _, c := range fx.TensorBytes {
		if got := TensorQuantizedBytes(c.Shape, c.Bits, c.GS, c.Mode); got != c.Bytes {
			t.Errorf("TensorQuantizedBytes(%v,%d,%d,%q) = %d, want %d", c.Shape, c.Bits, c.GS, c.Mode, got, c.Bytes)
		}
	}
}

func TestEstimateEffectiveBPWParity(t *testing.T) {
	fx := loadOQFixture(t)
	shapes := map[string][]int{"a.q_proj": {4096, 4096}, "a.down_proj": {11008, 4096}, "a.embed": {128000, 4096}}
	if got := EstimateEffectiveBPW(shapes, 4, 64, "affine", nil); !approxEq(got, fx.EffectiveBPW["plain"]) {
		t.Errorf("plain bpw = %v, want %v", got, fx.EffectiveBPW["plain"])
	}
	ov := map[string]map[string]any{"a.q_proj": {"bits": 6, "group_size": 64, "mode": "affine"}}
	if got := EstimateEffectiveBPW(shapes, 4, 64, "affine", ov); !approxEq(got, fx.EffectiveBPW["override"]) {
		t.Errorf("override bpw = %v, want %v", got, fx.EffectiveBPW["override"])
	}
}

func TestShouldQuantizeTensorParity(t *testing.T) {
	fx := loadOQFixture(t)
	for _, c := range fx.ShouldQuantize {
		if got := ShouldQuantizeTensor(c.Name, c.Shape); got != c.Q {
			t.Errorf("ShouldQuantizeTensor(%q,%v) = %v, want %v", c.Name, c.Shape, got, c.Q)
		}
	}
}

func TestShouldSkipTensorParity(t *testing.T) {
	fx := loadOQFixture(t)
	for _, c := range fx.ShouldSkip {
		if got := ShouldSkipTensor(c.Name, c.Preserve); got != c.Skip {
			t.Errorf("ShouldSkipTensor(%q,%v) = %v, want %v", c.Name, c.Preserve, got, c.Skip)
		}
	}
}

func TestIsMTPProtectedTensorParity(t *testing.T) {
	fx := loadOQFixture(t)
	for name, want := range fx.MTPProtected {
		if got := IsMTPProtectedTensor(name); got != want {
			t.Errorf("IsMTPProtectedTensor(%q) = %v, want %v", name, got, want)
		}
	}
}

func TestGetPredicateBitsParity(t *testing.T) {
	fx := loadOQFixture(t)
	for _, c := range fx.PredicateBits {
		bits, gs, mode, quantized := GetPredicateBits(c.Name, Config(c.Config), c.Level, c.GS)
		if string(c.Result[0]) == "null" {
			if quantized {
				t.Errorf("%q: expected not quantized, got (%d,%d,%q)", c.Name, bits, gs, mode)
			}
			continue
		}
		var wantBits, wantGS int
		var wantMode string
		json.Unmarshal(c.Result[0], &wantBits)
		json.Unmarshal(c.Result[1], &wantGS)
		json.Unmarshal(c.Result[2], &wantMode)
		if !quantized || bits != wantBits || gs != wantGS || mode != wantMode {
			t.Errorf("%q: got (%d,%d,%q) q=%v, want (%d,%d,%q)", c.Name, bits, gs, mode, quantized, wantBits, wantGS, wantMode)
		}
	}
}

func TestNormalizeMTPInConfig(t *testing.T) {
	cfg := Config{
		"mtp_num_hidden_layers":    2.0,
		"num_nextn_predict_layers": 1.0,
		"text_config":              map[string]any{"mtp_num_hidden_layers": 3.0},
	}
	NormalizeMTPInConfig(cfg)
	if cfg["mtp_num_hidden_layers"] != 0 || cfg["num_nextn_predict_layers"] != 0 {
		t.Errorf("top-level MTP counts not zeroed: %v", cfg)
	}
	if tc := cfg["text_config"].(map[string]any); tc["mtp_num_hidden_layers"] != 0 {
		t.Errorf("text_config MTP count not zeroed: %v", tc)
	}
}

func parseLevel(t *testing.T, s string) float64 {
	t.Helper()
	var f float64
	if err := json.Unmarshal([]byte(s), &f); err != nil {
		t.Fatalf("bad level %q: %v", s, err)
	}
	return f
}

func approxEq(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-9
}

func BenchmarkUniversalQuantPredicate(b *testing.B) {
	cfg := Config{"num_hidden_layers": 48.0, "num_local_experts": 128.0, "hidden_size": 4096.0}
	b.ReportAllocs()
	for b.Loop() {
		_ = UniversalQuantPredicate("model.layers.16.mlp.experts.3.down_proj.weight", cfg, 4)
	}
}
