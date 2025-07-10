# Makefile for building and cleaning language implementations

.PHONY: all clean rust swift typescript clean-rust clean-swift clean-typescript

# Default target: build all implementations
all: rust swift typescript

# Rust implementation
rust:
	@echo "Building Rust implementation..."
	cd rust && cargo build --release

clean-rust:
	@echo "Cleaning Rust implementation..."
	cd rust && cargo clean

# Swift implementation
swift:
	@echo "Building Swift implementation..."
	cd Swift && swift build -c release

clean-swift:
	@echo "Cleaning Swift implementation..."
	cd Swift && rm -rf .build

# TypeScript implementation
typescript:
	@echo "Building TypeScript implementation..."
	cd typescript && npm install

clean-typescript:
	@echo "Cleaning TypeScript implementation..."
	cd typescript && rm -rf node_modules

# Clean all implementations
clean: clean-rust clean-swift clean-typescript
	@echo "All implementations cleaned."

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build all implementations (default)"
	@echo "  rust             - Build Rust implementation"
	@echo "  swift            - Build Swift implementation"
	@echo "  typescript       - Build TypeScript implementation"
	@echo "  clean            - Clean all implementations"
	@echo "  clean-rust       - Clean Rust implementation"
	@echo "  clean-swift      - Clean Swift implementation"
	@echo "  clean-typescript - Clean TypeScript implementation"
