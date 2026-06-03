// SPDX-License-Identifier: MIT OR Apache-2.0

// Package install ports the installation-method detection: deciding whether the
// running binary came from the macOS app bundle, a Homebrew install, or a plain
// pip install, and from that the CLI command prefix a panel should show. The
// reference reads its own module path, sys.prefix, the home directory, and
// whether a shim file is executable; those live readings are the caller's seam,
// so each function here takes them as plain inputs.
package install

import (
	"path"
	"strings"
)

const (
	appBundleCLIName = "fastmlx-cli"
	pathCLI          = "fastmlx"
	defaultAppCLIDir = "/Applications/fastmlx.app/Contents/MacOS"
	appMarker        = ".app/Contents/"
	appSuffix        = ".app"
)

// IsAppBundle reports whether the given module path sits inside a macOS .app
// bundle, the marker the bundle build leaves on every packaged file.
func IsAppBundle(modulePath string) bool {
	return strings.Contains(modulePath, appMarker)
}

// AppBundleCLIPath returns the bundled CLI path for the bundle that contains the
// given module path, derived by trimming the path back to its ".app" root. A path
// with no bundle marker falls back to the default Applications location.
func AppBundleCLIPath(modulePath string) string {
	idx := strings.Index(modulePath, appMarker)
	if idx == -1 {
		return path.Join(defaultAppCLIDir, appBundleCLIName)
	}
	appRoot := modulePath[:idx+len(appSuffix)]
	return path.Join(appRoot, "Contents", "MacOS", appBundleCLIName)
}

// UserCLIShimPath returns the PATH shim the macOS app installs under the user's
// home directory.
func UserCLIShimPath(home string) string {
	return path.Join(home, ".fastmlx", "bin", "fastmlx")
}

// IsHomebrew reports whether the given interpreter prefix belongs to a Homebrew
// install, covering both the Intel/Apple-Silicon Cellar and the linuxbrew layout.
func IsHomebrew(prefix string) bool {
	return strings.Contains(prefix, "/Cellar/") || strings.Contains(prefix, "/homebrew/")
}

// InstallMethod classifies the install from the module path and interpreter
// prefix: "dmg" for the app bundle, then "homebrew", otherwise "pip".
func InstallMethod(modulePath, prefix string) string {
	if IsAppBundle(modulePath) {
		return "dmg"
	}
	if IsHomebrew(prefix) {
		return "homebrew"
	}
	return "pip"
}

// CLIPrefix returns the CLI command prefix for the current install. Outside an
// app bundle the bare PATH command is used. Inside a bundle the bare command is
// used when the user PATH shim is installed and executable, and the absolute
// bundled CLI path otherwise; whether the shim is executable is the caller's
// filesystem probe.
func CLIPrefix(modulePath string, userShimExecutable bool) string {
	if IsAppBundle(modulePath) {
		if userShimExecutable {
			return pathCLI
		}
		return AppBundleCLIPath(modulePath)
	}
	return pathCLI
}
