# SPDX-License-Identifier: MIT OR Apache-2.0

.PHONY: all build test race cover lint vet fmt fmtcheck vuln bench mlx build-mlx test-mlx clean

all: fmt vet lint test build

build:
	go build ./...

test:
	go test ./...

# race mirrors the CI gate: data-race detector + atomic coverage.
race:
	go test -race -covermode=atomic -coverprofile=coverage.out ./...

cover: race
	go tool cover -func=coverage.out | tail -1

vet:
	go vet ./...

# lint runs golangci-lint with the repo config (.golangci.yml).
lint:
	golangci-lint run ./...

fmt:
	gofmt -w .

fmtcheck:
	@unformatted=$$(gofmt -l .); \
	if [ -n "$$unformatted" ]; then echo "not gofmt-clean:"; echo "$$unformatted"; exit 1; fi

vuln:
	go run golang.org/x/vuln/cmd/govulncheck@latest ./...

# bench runs every benchmark once (smoke); raise -benchtime for real numbers.
bench:
	go test -run '^$$' -bench . -benchtime 1x ./...

# mlx builds the MLX + mlx-c dylibs the compute backend (v0.4+) links against.
# Populated when the third_party submodules land.
mlx:
	@echo "mlx-c bootstrap lands in v0.4 (spec 1990, 02_compute_backend_mlxc.md)"

# The default build compiles the GPU-free mlxgo stub, so every target above
# works without the MLX toolchain. build-mlx / test-mlx opt into the real cgo
# backend with -tags mlx; they require the MLX dylibs + headers (run `make mlx`
# first) and an Apple Silicon host with Metal.
build-mlx:
	go build -tags mlx ./...

test-mlx:
	go test -tags mlx ./...

clean:
	go clean ./...
	rm -f fastmlx coverage.out
