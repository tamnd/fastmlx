// SPDX-License-Identifier: MIT OR Apache-2.0

package profiles

import (
	"encoding/json"
	"maps"
)

// ApplyProfileToSettings reproduces the pure core of the settings manager's
// apply_profile: it overlays a profile's stored settings onto a model's current
// settings and records the profile as active. The current settings are taken to
// their to_dict form, the profile keys are layered on top, active_profile_name
// is set, and the result is rebuilt through the same defaults-and-validate path
// FromModelSettingsDict uses. A nil current starts from the defaults, matching
// the reference's fallback to a fresh ModelSettings. The disk write and lock in
// the manager are the seam left to the caller.
func ApplyProfileToSettings(current *ModelSettings, profileSettings map[string]any, name string) (ModelSettings, error) {
	base := NewModelSettings()
	if current != nil {
		base = *current
	}
	merged, err := settingsToMap(base)
	if err != nil {
		return ModelSettings{}, err
	}
	maps.Copy(merged, profileSettings)
	merged["active_profile_name"] = name
	data, err := json.Marshal(merged)
	if err != nil {
		return ModelSettings{}, err
	}
	return FromModelSettingsDict(data)
}

// settingsToMap renders a ModelSettings to its to_dict object as a generic map,
// so profile keys can be overlaid before the record is rebuilt.
func settingsToMap(m ModelSettings) (map[string]any, error) {
	data, err := m.ToDict()
	if err != nil {
		return nil, err
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// BuildProfileRecord reproduces the pure core of the manager's save_profile: it
// validates the name, filters the settings down to the profile-eligible fields,
// and assembles the serializable record. The display name falls back to the
// profile name when empty. Timestamps are passed in as the caller's clock seam
// (the reference uses utcnow().isoformat()); created_at and updated_at are both
// set to now on creation.
func BuildProfileRecord(name, displayName string, description *string, settings map[string]any, sourceTemplate *string, now string) (ModelProfile, error) {
	if err := ValidateProfileName(name); err != nil {
		return ModelProfile{}, err
	}
	if displayName == "" {
		displayName = name
	}
	return ModelProfile{
		Name:           name,
		DisplayName:    displayName,
		Description:    description,
		CreatedAt:      now,
		UpdatedAt:      now,
		Settings:       FilterProfileFields(settings),
		SourceTemplate: sourceTemplate,
	}, nil
}
