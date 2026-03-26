PYTHON ?= ./venv/bin/python

.PHONY: demo-up demo-down retrain push-neo4j demo-check logs restart-api

# Start all services
demo-up:
	docker compose up -d --build
	@echo ""
	@echo "Services starting..."
	@echo "  API:        http://localhost:8000"
	@echo "  API docs:   http://localhost:8000/docs"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3001 (admin/admin)"
	@echo ""
	@echo "Waiting for API and MLflow to be ready..."
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		if curl -sf http://localhost:8000/health > /dev/null && curl -sf http://localhost:5000 > /dev/null; then \
			echo "API and MLflow are ready ✅"; \
			exit 0; \
		fi; \
		sleep 2; \
	done; \
	echo "Services are still starting, try: make demo-check"

# Stop all services
demo-down:
	docker compose down

# Run training pipeline (runs on host, connects to Dockerized Redis/MLflow)
retrain:
	MLFLOW_ENABLED=true \
	MLFLOW_TRACKING_URI=http://localhost:5000 \
	MLFLOW_REGISTRY_ENABLED=true \
	MLFLOW_REGISTERED_MODEL_NAME=betblitz-lightfm-recsys \
	REDIS_ENABLED=true \
	REDIS_HOST=localhost \
	$(PYTHON) -m pipeline.run
	@echo ""
	@echo "Retraining complete. Restart API to pick up new artifacts:"
	@echo "  docker compose restart api"

# Push embeddings from the latest exported artifacts to Neo4j.
# Requires NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD to be exported in the shell.
push-neo4j:
	MLFLOW_ENABLED=false \
	REDIS_ENABLED=false \
	NEO4J_ENABLED=true \
	NEO4J_PLAYER_KEY=$${NEO4J_PLAYER_KEY:-id} \
	NEO4J_GAME_KEY=$${NEO4J_GAME_KEY:-id} \
	PYTHONPATH=. $(PYTHON) scripts/push_neo4j_embeddings.py
	@echo ""
	@echo "Neo4j push complete. Verify updates in Neo4j Browser:"
	@echo "  MATCH (p:Player) WHERE p.embedding IS NOT NULL RETURN count(p) AS players_with_embeddings"
	@echo "  MATCH (g:Game) WHERE g.embedding IS NOT NULL RETURN count(g) AS games_with_embeddings"

# Smoke test: health + known user + cold-start user
demo-check:
	@bash scripts/smoke_test.sh

# Tail logs
logs:
	docker compose logs -f

# Restart API (after retrain to pick up new artifacts)
restart-api:
	docker compose restart api
	@echo "API restarted with fresh artifacts ✅"
