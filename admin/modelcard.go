// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"fmt"
	"math"
	"strconv"
)

// GenerateModelCard builds the minimal Hugging Face model card written before an
// oQ-quantized model is uploaded. The card is pure given its inputs: the running
// version and today's date are passed in (the version lookup and the clock are
// caller seams), and the config is the already-parsed config.json. Missing
// fields fall back the way the reference dict lookups do: model_type to
// "unknown", and the quantization bits and group size to "?". The redownload
// notice block is included only when asked, carrying the supplied date.
func GenerateModelCard(modelName string, config map[string]any, redownloadNotice bool, today, version string) string {
	modelType := "unknown"
	if v, ok := config["model_type"]; ok {
		modelType = pyValueStr(v)
	}
	quant, _ := config["quantization"].(map[string]any)
	bits := "?"
	if v, ok := quant["bits"]; ok {
		bits = pyValueStr(v)
	}
	groupSize := "?"
	if v, ok := quant["group_size"]; ok {
		groupSize = pyValueStr(v)
	}

	notice := ""
	if redownloadNotice {
		notice = "> [!IMPORTANT]\n" +
			"> This quantization was uploaded on **" + today + "** and replaces a previous version.\n" +
			"> If you downloaded this model before this date, please re-download for the updated weights.\n\n"
	}

	return "---\n" +
		"library_name: mlx\n" +
		"tags:\n" +
		"- mlx\n" +
		"- oq\n" +
		"- quantized\n" +
		"---\n\n" +
		notice + "# " + modelName + "\n\n" +
		"This model was quantized using [oQ](https://github.com/tamnd/fastmlx) (fastmlx v" + version + ") mixed-precision quantization.\n\n" +
		"## Quantization details\n\n" +
		"- **Model type**: " + modelType + "\n" +
		"- **Bits**: " + bits + "\n" +
		"- **Group size**: " + groupSize + "\n" +
		"- **Format**: MLX safetensors\n"
}

// pyValueStr renders a JSON-decoded config value the way Python's str() would
// for the field-interpolation here: a string as itself, a whole number without a
// trailing ".0" (config bits and group sizes are integers), a fractional number
// in its shortest form, and a bool as True/False.
func pyValueStr(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case bool:
		if x {
			return "True"
		}
		return "False"
	case float64:
		if !math.IsInf(x, 0) && !math.IsNaN(x) && x == math.Trunc(x) {
			return strconv.FormatInt(int64(x), 10)
		}
		return strconv.FormatFloat(x, 'g', -1, 64)
	case nil:
		return "None"
	default:
		return fmt.Sprintf("%v", x)
	}
}
