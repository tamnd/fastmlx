// SPDX-License-Identifier: MIT OR Apache-2.0

package profiles

import (
	"encoding/json"
	"errors"
	"fmt"
)

// Settings file format versions.
const (
	SettingsVersion  = 1
	ProfilesVersion  = 1
	TemplatesVersion = 1
)

// ModelSettings holds the per-model configuration: sampling parameters,
// speculative-decoding toggles, management flags, and metadata. Every field
// that defaults to None in the reference is a pointer here, so an unset value
// stays absent from the serialized form rather than collapsing to a zero. The
// field order matches the reference dataclass declaration so the serialized
// object keys come out in the same order as to_dict.
type ModelSettings struct {
	MaxContextWindow   *int            `json:"max_context_window,omitempty"`
	MaxTokens          *int            `json:"max_tokens,omitempty"`
	Temperature        *float64        `json:"temperature,omitempty"`
	TopP               *float64        `json:"top_p,omitempty"`
	TopK               *int            `json:"top_k,omitempty"`
	RepetitionPenalty  *float64        `json:"repetition_penalty,omitempty"`
	MinP               *float64        `json:"min_p,omitempty"`
	PresencePenalty    *float64        `json:"presence_penalty,omitempty"`
	ForceSampling      bool            `json:"force_sampling"`
	MaxToolResultToken *int            `json:"max_tool_result_tokens,omitempty"`
	ChatTemplateKwargs *map[string]any `json:"chat_template_kwargs,omitempty"`
	ForcedCTKwargs     *[]string       `json:"forced_ct_kwargs,omitempty"`
	TTLSeconds         *int            `json:"ttl_seconds,omitempty"`
	ModelTypeOverride  *string         `json:"model_type_override,omitempty"`
	ModelAlias         *string         `json:"model_alias,omitempty"`
	IndexCacheFreq     *int            `json:"index_cache_freq,omitempty"`
	EnableThinking     *bool           `json:"enable_thinking,omitempty"`
	PreserveThinking   *bool           `json:"preserve_thinking,omitempty"`
	ThinkingBudgetEn   bool            `json:"thinking_budget_enabled"`
	ThinkingBudgetTok  *int            `json:"thinking_budget_tokens,omitempty"`
	ReasoningParser    *string         `json:"reasoning_parser,omitempty"`
	GuidedGrammarEn    bool            `json:"guided_grammar_enabled"`
	GuidedGrammar      *string         `json:"guided_grammar,omitempty"`

	TurboquantKVEnabled bool    `json:"turboquant_kv_enabled"`
	TurboquantKVBits    float64 `json:"turboquant_kv_bits"`
	TurboquantSkipLast  bool    `json:"turboquant_skip_last"`

	SpecprefillEnabled    bool     `json:"specprefill_enabled"`
	SpecprefillDraftModel *string  `json:"specprefill_draft_model,omitempty"`
	SpecprefillKeepPct    *float64 `json:"specprefill_keep_pct,omitempty"`
	SpecprefillThreshold  *int     `json:"specprefill_threshold,omitempty"`

	DflashEnabled              bool    `json:"dflash_enabled"`
	DflashDraftModel           *string `json:"dflash_draft_model,omitempty"`
	DflashDraftQuantEnabled    *bool   `json:"dflash_draft_quant_enabled,omitempty"`
	DflashDraftQuantWeightBits *int    `json:"dflash_draft_quant_weight_bits,omitempty"`
	DflashDraftQuantActBits    *int    `json:"dflash_draft_quant_activation_bits,omitempty"`
	DflashDraftQuantGroupSize  *int    `json:"dflash_draft_quant_group_size,omitempty"`
	DflashMaxCtx               *int    `json:"dflash_max_ctx,omitempty"`
	DflashInMemoryCache        bool    `json:"dflash_in_memory_cache"`
	DflashInMemoryCacheEntries int     `json:"dflash_in_memory_cache_max_entries"`
	DflashInMemoryCacheBytes   int64   `json:"dflash_in_memory_cache_max_bytes"`
	DflashSSDCache             bool    `json:"dflash_ssd_cache"`
	DflashSSDCacheBytes        int64   `json:"dflash_ssd_cache_max_bytes"`
	DflashDraftWindowSize      *int    `json:"dflash_draft_window_size,omitempty"`
	DflashDraftSinkSize        *int    `json:"dflash_draft_sink_size,omitempty"`
	DflashVerifyMode           *string `json:"dflash_verify_mode,omitempty"`

	MTPEnabled bool `json:"mtp_enabled"`

	VLMMTPEnabled      bool    `json:"vlm_mtp_enabled"`
	VLMMTPDraftModel   *string `json:"vlm_mtp_draft_model,omitempty"`
	VLMMTPDraftBlockSz *int    `json:"vlm_mtp_draft_block_size,omitempty"`

	IsPinned        bool `json:"is_pinned"`
	IsDefault       bool `json:"is_default"`
	TrustRemoteCode bool `json:"trust_remote_code"`

	DisplayName       *string `json:"display_name,omitempty"`
	Description       *string `json:"description,omitempty"`
	ActiveProfileName *string `json:"active_profile_name,omitempty"`
}

// NewModelSettings returns a ModelSettings with the reference's non-None
// defaults applied. The remaining fields default to nil or the bool/zero value,
// matching the dataclass.
func NewModelSettings() ModelSettings {
	return ModelSettings{
		TurboquantKVBits:           4,
		TurboquantSkipLast:         true,
		DflashInMemoryCache:        true,
		DflashInMemoryCacheEntries: 4,
		DflashInMemoryCacheBytes:   8 * 1024 * 1024 * 1024,
		DflashSSDCacheBytes:        20 * 1024 * 1024 * 1024,
	}
}

// Validate reproduces the reference __post_init__ checks: the speculative
// decoding paths are mutually exclusive. It returns the first conflict found,
// with the same message the reference raises, or nil when the combination is
// allowed.
func (m ModelSettings) Validate() error {
	// Native MTP conflicts with DFlash and TurboQuant, which patch the same
	// attention path.
	if m.MTPEnabled && m.DflashEnabled {
		return errors.New("mtp_enabled and dflash_enabled cannot both be True; choose one speculative-decoding path per model")
	}
	if m.MTPEnabled && m.TurboquantKVEnabled {
		return errors.New("mtp_enabled and turboquant_kv_enabled cannot both be True; TurboQuant patches the attention path that MTP relies on")
	}
	// VLM MTP wraps the mlx-vlm loop and cannot coexist with any other
	// speculative path or with TurboQuant. The conflict order matches the
	// reference so the reported field is identical.
	if m.VLMMTPEnabled {
		conflicts := []struct {
			name string
			on   bool
		}{
			{"dflash_enabled", m.DflashEnabled},
			{"specprefill_enabled", m.SpecprefillEnabled},
			{"mtp_enabled", m.MTPEnabled},
			{"turboquant_kv_enabled", m.TurboquantKVEnabled},
		}
		for _, c := range conflicts {
			if c.on {
				return fmt.Errorf("vlm_mtp_enabled and %s cannot both be True; choose one speculative path per model", c.name)
			}
		}
	}
	return nil
}

// ToDict serializes the settings the way the reference to_dict does: every
// None-valued field is dropped and the rest are emitted in declaration order.
// The omitempty pointer fields cover the None exclusion; the non-pointer fields
// (which always carry a concrete default in the reference) are always present.
func (m ModelSettings) ToDict() ([]byte, error) {
	return json.Marshal(m)
}

// FromModelSettingsDict builds a ModelSettings from a serialized object. It
// starts from the defaults, applies the supplied keys, and ignores any key that
// is not a known field (matching the reference's filter to valid field names).
// The mutual-exclusion rules are enforced, so an invalid combination is
// reported rather than silently constructed.
func FromModelSettingsDict(data []byte) (ModelSettings, error) {
	m := NewModelSettings()
	if err := json.Unmarshal(data, &m); err != nil {
		return ModelSettings{}, err
	}
	if err := m.Validate(); err != nil {
		return ModelSettings{}, err
	}
	return m, nil
}
