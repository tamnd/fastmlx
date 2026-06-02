// SPDX-License-Identifier: Apache-2.0

// Command fastmlx is the fastmlx CLI: a continuous-batching
// MLX LLM inference server for Apple Silicon. It provides serve/launch/diagnose
// subcommands and reads the ~/.fastmlx configuration.
package main

import (
	"fmt"
	"os"
)

const version = "0.1.0-dev"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	cmd := os.Args[1]
	args := os.Args[2:]
	switch cmd {
	case "serve":
		if err := runServe(args); err != nil {
			fmt.Fprintln(os.Stderr, "fastmlx serve:", err)
			os.Exit(1)
		}
	case "launch":
		if err := runLaunch(args); err != nil {
			fmt.Fprintln(os.Stderr, "fastmlx launch:", err)
			os.Exit(1)
		}
	case "diagnose":
		if err := runDiagnose(args); err != nil {
			fmt.Fprintln(os.Stderr, "fastmlx diagnose:", err)
			os.Exit(1)
		}
	case "version", "--version", "-v":
		fmt.Println("fastmlx", version)
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", cmd)
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `fastmlx - LLM inference server, optimized for your Mac

Usage:
  fastmlx <command> [options]

Commands:
  serve       Start the inference server
  launch      Launch a coding agent pointed at the local server
  diagnose    Diagnostics (e.g. menubar)
  version     Print version

Run "fastmlx <command> --help" for command-specific options.
`)
}
