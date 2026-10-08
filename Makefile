# Ava Multimodal WhatsApp AI Agent

ifeq (,$(wildcard .env))
$(error .env file is missing. Please create one based on .env.example)
endif

CHECK_DIRS := .

.PHONY: help ava-build ava-run ava-run-chainlit ava-run-whatsapp \
        ava-stop ava-restart ava-delete ava-status \
        ava-logs ava-logs-chainlit ava-logs-whatsapp ava-logs-qdrant \
        dev-chainlit dev-whatsapp \
        format-fix lint-fix format-check lint-check

# Default
help: ## Show all available targets
	@echo ""
	@echo "Ava Multimodal AI Agent"
	@echo "  ────────────────────────────────────────────"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""

# Full stack
ava-build: ## Build all Docker images (with cache)
	docker compose build

ava-run: ## Start ALL services (Qdrant + Chainlit + WhatsApp)
	docker compose up --build -d
	@echo ""
	@echo "  All services started"
	@echo "  ─────────────────────────────────────"
	@echo "  Chainlit UI   → http://localhost:8000"
	@echo "  WhatsApp API  → http://localhost:8080"
	@echo "  Qdrant        → http://localhost:6333"
	@echo ""

ava-run-chainlit: ## Start Qdrant + Chainlit only
	docker compose up --build -d qdrant chainlit
	@echo ""
	@echo "Chainlit started → http://localhost:8000"
	@echo ""

ava-run-whatsapp: ## Start Qdrant + WhatsApp only
	docker compose up --build -d qdrant whatsapp
	@echo ""
	@echo "WhatsApp API started → http://localhost:8080"
	@echo ""

# Lifecycle
ava-stop: ## Stop all containers
	docker compose stop

ava-restart: ava-stop ava-run ## Restart all services

ava-status: ## Show running containers
	docker compose ps

# Logs
ava-logs: ## Tail logs for all services
	docker compose logs -f

ava-logs-chainlit: ## Tail Chainlit logs
	docker compose logs -f chainlit

ava-logs-whatsapp: ## Tail WhatsApp logs
	docker compose logs -f whatsapp

ava-logs-qdrant: ## Tail Qdrant logs
	docker compose logs -f qdrant

# Cleanup
ava-delete: ## Stop everything + remove containers and data dirs
	docker compose down
	@if [ -d "long_term_memory" ]; then rm -rf long_term_memory; fi
	@if [ -d "short_term_memory" ]; then rm -rf short_term_memory; fi
	@if [ -d "generated_images" ]; then rm -rf generated_images; fi
	@echo "Cleaned up containers and data."

# Local dev (no Docker for app, only Qdrant)
dev-chainlit: ## Run Chainlit locally (Qdrant in Docker)
	docker compose up -d qdrant
	@echo "Waiting for Qdrant healthcheck…"
	@until docker compose exec qdrant curl -sf http://localhost:6333/healthz > /dev/null 2>&1; do sleep 1; done
	@echo "Qdrant ready — starting Chainlit…"
	QDRANT_URL=http://localhost:6333 uv run chainlit run \
		src/ai_companion/interfaces/chainlit/app.py --port 8000

dev-whatsapp: ## Run WhatsApp server locally (Qdrant in Docker)
	docker compose up -d qdrant
	@echo "Waiting for Qdrant healthcheck…"
	@until docker compose exec qdrant curl -sf http://localhost:6333/healthz > /dev/null 2>&1; do sleep 1; done
	@echo "Qdrant ready — starting WhatsApp server…"
	QDRANT_URL=http://localhost:6333 uv run fastapi dev \
		src/ai_companion/interfaces/whatsapp/webhook_endpoint.py --port 8080

# Code quality
format-fix: ## Auto-format code with ruff
	uv run ruff format $(CHECK_DIRS)
	uv run ruff check --select I --fix $(CHECK_DIRS)

lint-fix: ## Auto-fix lint issues
	uv run ruff check --fix $(CHECK_DIRS)

format-check: ## Check formatting (CI-friendly)
	uv run ruff format --check $(CHECK_DIRS)
	uv run ruff check -e $(CHECK_DIRS)
	uv run ruff check --select I -e $(CHECK_DIRS)

lint-check: ## Check for lint errors
	uv run ruff check $(CHECK_DIRS)



