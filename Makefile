# ai-elevenlabs-analysis — Local Development Makefile
# Usage: make help

SHELL := /bin/bash
PYTHON ?= python3
BASELINE ?= 0.05
Z_THRESHOLD ?= 3.0
DATA_DIR := data
SCRIPTS_DIR := scripts
ANALYSIS_DIR := analysis

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m

.PHONY: help analyze analyze-json deep-dive lint format typecheck validate compile-c clean all

help: ## Show this help
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║           AI ElevenLabs Music Analysis — Makefile           ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-18s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "  Variables:"
	@echo "    BASELINE=$(BASELINE)       Override: make analyze BASELINE=0.03"
	@echo "    Z_THRESHOLD=$(Z_THRESHOLD)    Override: make analyze Z_THRESHOLD=2.5"

all: validate lint analyze deep-dive ## Run full pipeline (validate → lint → analyze → deep-dive)

# ─── Analysis ────────────────────────────────────────────────────

analyze: ## Run void cluster analysis on all data files (markdown output)
	@echo "$(GREEN)═══ Void Cluster Analysis ═══$(NC)"
	@for f in $(DATA_DIR)/*.md $(DATA_DIR)/*.txt; do \
		[ -f "$$f" ] || continue; \
		echo ""; \
		echo "$(YELLOW)▶ $$f$(NC)"; \
		$(PYTHON) $(SCRIPTS_DIR)/analyze.py "$$f" --baseline $(BASELINE) --format markdown; \
	done

analyze-json: ## Run analysis with JSON output (for piping/processing)
	@for f in $(DATA_DIR)/*.md $(DATA_DIR)/*.txt; do \
		[ -f "$$f" ] || continue; \
		echo "--- $$f ---"; \
		$(PYTHON) $(SCRIPTS_DIR)/analyze.py "$$f" --baseline $(BASELINE) --format json; \
	done

analyze-check: ## Run analysis and check z-scores against threshold
	@echo "$(GREEN)═══ Threshold Check (z > $(Z_THRESHOLD)) ═══$(NC)"
	@alert=0; \
	for f in $(DATA_DIR)/*.md $(DATA_DIR)/*.txt; do \
		[ -f "$$f" ] || continue; \
		z=$$($(PYTHON) $(SCRIPTS_DIR)/analyze.py "$$f" --baseline $(BASELINE) --format json 2>/dev/null | \
			$(PYTHON) -c "import sys,json; d=json.load(sys.stdin); t=d.get('statistical_tests',{}); print(t.get('dark_prog',t.get('custom',{'z_score':0}))['z_score'])" 2>/dev/null || echo "0"); \
		exceeds=$$($(PYTHON) -c "print('YES' if float('$$z') > $(Z_THRESHOLD) else 'no')"); \
		if [ "$$exceeds" = "YES" ]; then \
			echo "$(RED)🚨 $$f: z=$$z EXCEEDS $(Z_THRESHOLD)$(NC)"; \
			alert=1; \
		else \
			echo "  $$f: z=$$z — ok"; \
		fi; \
	done; \
	if [ "$$alert" -eq 1 ]; then \
		echo ""; \
		echo "$(RED)⚠ One or more files exceeded the z-score threshold$(NC)"; \
	else \
		echo ""; \
		echo "$(GREEN)✅ All files within threshold$(NC)"; \
	fi

deep-dive: ## Run deep dive analyzer (Shadows of Geometry)
	@echo "$(GREEN)═══ Deep Dive Analysis ═══$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/deep_dive_analyzer.py

# ─── Validation & Quality ───────────────────────────────────────

validate: ## Validate data file format and encoding
	@echo "$(GREEN)═══ Data Validation ═══$(NC)"
	@errors=0; \
	for f in $(DATA_DIR)/*.md $(DATA_DIR)/*.txt; do \
		[ -f "$$f" ] || continue; \
		if [ ! -s "$$f" ]; then \
			echo "$(RED)✗ $$f — empty file$(NC)"; \
			errors=$$((errors + 1)); \
		elif file "$$f" | grep -q "UTF-8\|ASCII\|text"; then \
			echo "  ✓ $$f"; \
		else \
			echo "$(YELLOW)? $$f — unusual encoding$(NC)"; \
		fi; \
	done; \
	if [ "$$errors" -gt 0 ]; then \
		echo "$(RED)$$errors error(s) found$(NC)"; \
		exit 1; \
	fi; \
	echo "$(GREEN)✅ All data files valid$(NC)"

lint: ## Lint Python scripts with ruff
	@echo "$(GREEN)═══ Linting ═══$(NC)"
	@command -v ruff >/dev/null 2>&1 && ruff check $(SCRIPTS_DIR)/ || \
		echo "$(YELLOW)⚠ ruff not installed. Run: pip install ruff$(NC)"

format: ## Auto-format Python scripts with ruff
	@command -v ruff >/dev/null 2>&1 && ruff format $(SCRIPTS_DIR)/ || \
		echo "$(YELLOW)⚠ ruff not installed. Run: pip install ruff$(NC)"

typecheck: ## Run mypy type checking (informational)
	@command -v mypy >/dev/null 2>&1 && mypy $(SCRIPTS_DIR)/*.py --ignore-missing-imports || \
		echo "$(YELLOW)⚠ mypy not installed. Run: pip install mypy$(NC)"

compile-c: ## Compile C analysis tools
	@echo "$(GREEN)═══ Compiling C tools ═══$(NC)"
	@for f in $(SCRIPTS_DIR)/*.c; do \
		[ -f "$$f" ] || continue; \
		out="$${f%.c}"; \
		echo "  gcc $$f → $$out"; \
		gcc -Wall -Wextra -O2 -lm -o "$$out" "$$f"; \
	done
	@echo "$(GREEN)✅ Done$(NC)"

# ─── Utilities ───────────────────────────────────────────────────

clean: ## Remove compiled artifacts and caches
	@echo "Cleaning..."
	@rm -f $(SCRIPTS_DIR)/void-cluster-analyzer
	@rm -rf __pycache__ $(SCRIPTS_DIR)/__pycache__ .mypy_cache .ruff_cache
	@echo "$(GREEN)✅ Clean$(NC)"

word-count: ## Count total tokens across all data files
	@total=0; \
	for f in $(DATA_DIR)/*.md $(DATA_DIR)/*.txt; do \
		[ -f "$$f" ] || continue; \
		wc=$$(wc -w < "$$f"); \
		echo "  $$f: $$wc words"; \
		total=$$((total + wc)); \
	done; \
	echo "  Total: $$total words"

setup: ## Install development dependencies
	@echo "$(GREEN)═══ Setup ═══$(NC)"
	pip install ruff mypy
	@echo "$(GREEN)✅ Dev tools installed$(NC)"
