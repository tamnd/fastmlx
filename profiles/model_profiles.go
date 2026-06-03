// SPDX-License-Identifier: MIT OR Apache-2.0

// Package profiles holds the per-model settings profile and global template
// primitives: the field allowlists that split settings into universal,
// model-specific, and excluded groups; the serializable profile and template
// records; and profile-name validation.
package profiles

import (
	"fmt"
	"maps"
	"regexp"
)

// UniversalProfileFields are eligible for both global templates and per-model
// profiles.
var UniversalProfileFields = []string{
	"max_context_window",
	"max_tokens",
	"temperature",
	"top_p",
	"top_k",
	"min_p",
	"repetition_penalty",
	"presence_penalty",
	"force_sampling",
	"enable_thinking",
	"preserve_thinking",
	"thinking_budget_enabled",
	"thinking_budget_tokens",
	"reasoning_parser",
	"guided_grammar_enabled",
	"guided_grammar",
	"max_tool_result_tokens",
	"chat_template_kwargs",
	"forced_ct_kwargs",
}

// ModelSpecificProfileFields are eligible for per-model profiles only, never
// global templates.
var ModelSpecificProfileFields = []string{
	"turboquant_kv_enabled",
	"turboquant_kv_bits",
	"turboquant_skip_last",
	"dflash_enabled",
	"dflash_draft_model",
	"dflash_draft_quant_enabled",
	"dflash_draft_quant_weight_bits",
	"dflash_draft_quant_activation_bits",
	"dflash_draft_quant_group_size",
	"dflash_max_ctx",
	"dflash_in_memory_cache",
	"dflash_in_memory_cache_max_entries",
	"dflash_in_memory_cache_max_bytes",
	"dflash_ssd_cache",
	"dflash_ssd_cache_max_bytes",
	"dflash_draft_window_size",
	"dflash_draft_sink_size",
	"dflash_verify_mode",
	"mtp_enabled",
	"vlm_mtp_enabled",
	"vlm_mtp_draft_model",
	"vlm_mtp_draft_block_size",
	"specprefill_enabled",
	"specprefill_draft_model",
	"specprefill_keep_pct",
	"specprefill_threshold",
	"index_cache_freq",
}

// ExcludedFromProfiles are identity and management fields never stored in a
// profile or template. trust_remote_code is excluded deliberately: the security
// flag must be set explicitly per model, never propagated through a profile.
var ExcludedFromProfiles = map[string]bool{
	"is_pinned":           true,
	"is_default":          true,
	"display_name":        true,
	"description":         true,
	"model_alias":         true,
	"model_type_override": true,
	"active_profile_name": true,
	"ttl_seconds":         true,
	"trust_remote_code":   true,
}

// FilterUniversalFields returns a new map with only the universal field keys.
func FilterUniversalFields(data map[string]any) map[string]any {
	return filterByAllowed(data, allowedUniversal)
}

// FilterProfileFields returns a new map with the universal and model-specific
// keys.
func FilterProfileFields(data map[string]any) map[string]any {
	return filterByAllowed(data, allowedProfile)
}

var (
	allowedUniversal = makeSet(UniversalProfileFields)
	allowedProfile   = makeSet(append(append([]string{}, UniversalProfileFields...), ModelSpecificProfileFields...))
)

func makeSet(keys []string) map[string]bool {
	s := make(map[string]bool, len(keys))
	for _, k := range keys {
		s[k] = true
	}
	return s
}

func filterByAllowed(data map[string]any, allowed map[string]bool) map[string]any {
	out := map[string]any{}
	for k, v := range data {
		if allowed[k] {
			out[k] = v
		}
	}
	return out
}

// ModelProfile is a per-model saved bundle of settings values. Timestamps are
// carried as ISO 8601 strings (the reference's datetime.isoformat output).
type ModelProfile struct {
	Name           string         `json:"name"`
	DisplayName    string         `json:"display_name"`
	Description    *string        `json:"description"`
	CreatedAt      string         `json:"created_at"`
	UpdatedAt      string         `json:"updated_at"`
	Settings       map[string]any `json:"settings"`
	SourceTemplate *string        `json:"source_template"`
}

// GlobalTemplate is a globally shared bundle of universal settings values.
type GlobalTemplate struct {
	Name        string         `json:"name"`
	DisplayName string         `json:"display_name"`
	Description *string        `json:"description"`
	CreatedAt   string         `json:"created_at"`
	UpdatedAt   string         `json:"updated_at"`
	Settings    map[string]any `json:"settings"`
}

// normalizeSettings returns a non-nil copy so an absent settings map serializes
// as {} rather than null, matching the reference's `dict(data or {})`.
func normalizeSettings(s map[string]any) map[string]any {
	out := make(map[string]any, len(s))
	maps.Copy(out, s)
	return out
}

// Normalize ensures the settings map is non-nil. Decoders should call it after
// unmarshaling so a missing settings field becomes an empty object.
func (p *ModelProfile) Normalize() { p.Settings = normalizeSettings(p.Settings) }

// Normalize ensures the settings map is non-nil.
func (g *GlobalTemplate) Normalize() { g.Settings = normalizeSettings(g.Settings) }

// InvalidProfileNameError is returned when a profile or template name fails
// validation.
type InvalidProfileNameError struct {
	Name string
}

func (e *InvalidProfileNameError) Error() string {
	return fmt.Sprintf("invalid profile/template name: %q. Must match ^[a-z0-9][a-z0-9_-]{0,31}$", e.Name)
}

// nameRE matches a valid profile/template slug: lowercase letters or digits,
// underscores, and dashes; starts with a letter or digit; 1 to 32 characters.
var nameRE = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{0,31}$`)

// ValidateProfileName returns an InvalidProfileNameError when name is not a
// valid slug.
func ValidateProfileName(name string) error {
	if !nameRE.MatchString(name) {
		return &InvalidProfileNameError{Name: name}
	}
	return nil
}
