// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"strconv"
	"strings"
)

// This file ports the tool-schema preparation step that runs before a prompt is
// built: normalizing OpenAI-style tool definitions into the shape chat templates
// expect, and the Gemma 4 workarounds that rename parameters colliding with JSON
// Schema keywords and backfill missing parameter descriptions. These are pure
// structural transforms over the order-preserving jval model, so object key
// order is preserved exactly for byte-faithful prompt rendering.

// gemma4CollidingParams are parameter names Gemma 4 confuses with schema-level
// fields and drops from its tool-call output; they are renamed before the chat
// template and restored after parsing. gemma4RenamePrefix is the prefix applied.
var gemma4CollidingParams = map[string]bool{"description": true}

const gemma4RenamePrefix = "param_"

// pythonTruthy reports whether a jval is truthy under Python's rules: null,
// false, empty string, zero, and empty containers are falsy.
func pythonTruthy(v jval) bool {
	switch v.kind {
	case kindNull:
		return false
	case kindBool:
		return v.s == "true"
	case kindString:
		return v.s != ""
	case kindNumber:
		f, err := strconv.ParseFloat(v.s, 64)
		if err != nil {
			return true
		}
		return f != 0
	case kindObject:
		return len(v.obj) > 0
	case kindArray:
		return len(v.arr) > 0
	}
	return false
}

// getOr returns the named field if present, otherwise the supplied default,
// mirroring Python's dict.get(key, default).
func (v jval) getOr(key string, def jval) jval {
	if f, ok := v.getField(key); ok {
		return f
	}
	return def
}

// ConvertToolsForTemplate normalizes OpenAI tool definitions into the
// {type, function:{name, description, parameters}} shape chat templates expect.
// It keeps only function tools with a truthy function body, defaults a missing
// name/description to the empty string and missing parameters to an empty object
// schema, and returns a JSON null when there are no resulting tools.
func ConvertToolsForTemplate(tools jval) jval {
	if !pythonTruthy(tools) || tools.kind != kindArray {
		return jval{kind: kindNull}
	}
	defaultParams := jobj("type", jstr("object"), "properties", jval{kind: kindObject})
	var converted []jval
	for _, tool := range tools.arr {
		if tool.kind != kindObject {
			continue
		}
		toolType := tool.getString("type")
		toolFunc, ok := tool.getField("function")
		if toolType != "function" || !ok || !pythonTruthy(toolFunc) || toolFunc.kind != kindObject {
			continue
		}
		converted = append(converted, jobj(
			"type", jstr("function"),
			"function", jobj(
				"name", toolFunc.getOr("name", jstr("")),
				"description", toolFunc.getOr("description", jstr("")),
				"parameters", toolFunc.getOr("parameters", defaultParams),
			),
		))
	}
	if len(converted) == 0 {
		return jval{kind: kindNull}
	}
	return jval{kind: kindArray, arr: converted}
}

// EnrichToolParamsForGemma4 applies the two Gemma 4 schema fixups: it renames
// parameters whose names collide with JSON Schema keywords (description ->
// param_description) and synthesizes a description for any parameter that lacks
// one, marking required parameters with a "REQUIRED. " prefix. The required list
// is rebuilt to follow the (possibly renamed) property order. Use
// RestoreGemma4ParamNames to reverse the renaming on parsed arguments.
func EnrichToolParamsForGemma4(tools jval) jval {
	enriched := make([]jval, 0, len(tools.arr))
	for _, tool := range tools.arr {
		func_ := tool.getOr("function", jval{kind: kindObject})
		params := func_.getOr("parameters", jval{kind: kindObject})
		if params.kind == kindObject && params.hasField("properties") {
			oldProps := params.getOr("properties", jval{kind: kindObject})
			required := requiredSet(params)
			var newProps []jkv
			var newRequired []jval
			for _, kv := range oldProps.obj {
				pname := kv.k
				pdef := kv.v
				newName := pname
				if gemma4CollidingParams[pname] {
					newName = gemma4RenamePrefix + pname
				}
				if desc, ok := pdef.getField("description"); !ok || !pythonTruthy(desc) {
					label := ""
					if required[pname] {
						label = "REQUIRED. "
					}
					typeStr := "string"
					if t, ok := pdef.getField("type"); ok && t.kind == kindString {
						typeStr = t.s
					}
					text := label + "The '" + pname + "' value (type: " + typeStr + ")"
					pdef = pdef.setField("description", jstr(text))
				}
				newProps = append(newProps, jkv{newName, pdef})
				if required[pname] {
					newRequired = append(newRequired, jstr(newName))
				}
			}
			params = params.setField("properties", jval{kind: kindObject, obj: newProps})
			params = params.setField("required", jval{kind: kindArray, arr: newRequired})
			func_ = func_.setField("parameters", params)
			tool = tool.setField("function", func_)
		}
		enriched = append(enriched, tool)
	}
	return jval{kind: kindArray, arr: enriched}
}

// requiredSet collects the string entries of a schema's "required" array.
func requiredSet(params jval) map[string]bool {
	set := map[string]bool{}
	if req, ok := params.getField("required"); ok && req.kind == kindArray {
		for _, r := range req.arr {
			if r.kind == kindString {
				set[r.s] = true
			}
		}
	}
	return set
}

// RestoreGemma4ParamNames reverses the renaming done by
// EnrichToolParamsForGemma4: a key prefixed with "param_" whose remainder is a
// colliding name is restored to that name; every other key passes through.
func RestoreGemma4ParamNames(arguments jval) jval {
	out := jval{kind: kindObject}
	for _, kv := range arguments.obj {
		k := kv.k
		if strings.HasPrefix(k, gemma4RenamePrefix) {
			original := k[len(gemma4RenamePrefix):]
			if gemma4CollidingParams[original] {
				out.obj = append(out.obj, jkv{original, kv.v})
				continue
			}
		}
		out.obj = append(out.obj, jkv{k, kv.v})
	}
	return out
}

// FormatToolCallForMessage renders a ToolCall as the dict used when embedding it
// back into a message's content.
func FormatToolCallForMessage(tc ToolCall) jval {
	return jobj(
		"id", jstr(tc.ID),
		"type", jstr(tc.Type),
		"function", jobj(
			"name", jstr(tc.Function.Name),
			"arguments", jstr(tc.Function.Arguments),
		),
	)
}
