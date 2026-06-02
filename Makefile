# SPDX-License-Identifier: Apache-2.0

.PHONY: build test vet fmt lint all mlx clean

all: fmt vet test build

build:
	go build ./...

test:
	go test ./...

vet:
	go vet ./...

fmt:
	gofmt -w .

# mlx builds the MLX + mlx-c dylibs the compute backend (v0.4+) links against.
# Populated when the third_party submodules land.
mlx:
	@echo "mlx-c bootstrap lands in v0.4 (spec 1990, 02_compute_backend_mlxc.md)"

clean:
	go clean ./...
	rm -f fastmlx
