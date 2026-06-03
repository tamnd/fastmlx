// SPDX-License-Identifier: MIT OR Apache-2.0

package profiles

// BuildTemplateRecord reproduces the pure core of the settings manager's
// save_template: it validates the name, filters the settings down to the
// universal (template-eligible) fields, and assembles a fresh GlobalTemplate.
// The display name falls back to the template name when empty, and both
// timestamps are set to the injected clock value (the reference uses
// utcnow().isoformat()). The "template already exists" guard needs the template
// map and so is the caller's seam, alongside the lock and disk write.
func BuildTemplateRecord(name, displayName string, description *string, settings map[string]any, now string) (GlobalTemplate, error) {
	if err := ValidateProfileName(name); err != nil {
		return GlobalTemplate{}, err
	}
	if displayName == "" {
		displayName = name
	}
	return GlobalTemplate{
		Name:        name,
		DisplayName: displayName,
		Description: description,
		CreatedAt:   now,
		UpdatedAt:   now,
		Settings:    FilterUniversalFields(settings),
	}, nil
}

// UpsertTemplateRecord reproduces the pure core of upsert_template: create or
// replace a template, preserving an existing record's created_at while always
// stamping updated_at with the new clock value. Pass existing as nil to create.
// As with BuildTemplateRecord, the lock, map, and disk write are the caller's.
func UpsertTemplateRecord(name, displayName string, description *string, settings map[string]any, existing *GlobalTemplate, now string) (GlobalTemplate, error) {
	if err := ValidateProfileName(name); err != nil {
		return GlobalTemplate{}, err
	}
	if displayName == "" {
		displayName = name
	}
	createdAt := now
	if existing != nil {
		createdAt = existing.CreatedAt
	}
	return GlobalTemplate{
		Name:        name,
		DisplayName: displayName,
		Description: description,
		CreatedAt:   createdAt,
		UpdatedAt:   now,
		Settings:    FilterUniversalFields(settings),
	}, nil
}

// UpdateTemplateRecord reproduces the pure core of update_template: an
// optional-field overlay onto an existing record. A nil pointer leaves that
// field unchanged; a non-nil newName different from the current name is
// validated and applied (the returned record's Name carries the rename target).
// updated_at is always stamped with now. The not-found lookup and the
// rename-collision check both need the template map and so are the caller's
// seam, as are the lock and disk write.
func UpdateTemplateRecord(current GlobalTemplate, newName, displayName, description *string, settings *map[string]any, now string) (GlobalTemplate, error) {
	updated := current
	if newName != nil && *newName != updated.Name {
		if err := ValidateProfileName(*newName); err != nil {
			return GlobalTemplate{}, err
		}
		updated.Name = *newName
	}
	if displayName != nil {
		updated.DisplayName = *displayName
	}
	if description != nil {
		updated.Description = description
	}
	if settings != nil {
		updated.Settings = FilterUniversalFields(*settings)
	}
	updated.UpdatedAt = now
	return updated, nil
}
