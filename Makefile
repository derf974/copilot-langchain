.PHONY: help install test integration-test lint format clean

help: ## Afficher cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Installer les dépendances
	uv sync --all-extras --dev

test: ## Exécuter les tests unitaires
	uv run pytest tests/unit_tests/ -v

integration-test: ## Exécuter les tests d'intégration (nécessite Copilot CLI)
	uv run pytest tests/integration_tests/ -v
	
test-all: ## Exécuter tous les tests (unitaires + intégration)
	uv run pytest tests/ -v

lint: ## Vérifier le code avec ruff
	uv run ruff check .

format: ## Formater le code avec black et ruff
	uv run black .
	uv run ruff check --fix .

clean: ## Nettoyer les fichiers générés
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
