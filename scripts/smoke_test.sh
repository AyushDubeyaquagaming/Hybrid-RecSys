#!/usr/bin/env bash
set -e

API="http://localhost:8000"

echo "=== BetBlitz Smoke Test ==="
echo ""

# 1. Health check
echo "1. Health check..."
HEALTH=$(curl -sf "$API/health")
echo "   $HEALTH"
echo ""

# 2. Known user recommendation
echo "2. Known user recommendation..."
FIRST_USER=$(python3 -c "
import json
with open('betblitz-recsys-api/artifacts/user_id_map.json') as f:
    users = json.load(f)
print(next(iter(users.keys())))
" 2>/dev/null || echo "1244444444")

RECS=$(curl -sf -X POST "$API/recommend" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$FIRST_USER\", \"top_k\": 5}")
echo "   User: $FIRST_USER"
echo "   $RECS" | python3 -m json.tool 2>/dev/null || echo "   $RECS"
echo ""

# 3. Cold-start user
echo "3. Cold-start user (popularity fallback)..."
COLD=$(curl -sf -X POST "$API/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "unknown_demo_user", "top_k": 5}')
echo "   $COLD" | python3 -m json.tool 2>/dev/null || echo "   $COLD"
echo ""

# 4. Metrics endpoint
echo "4. Prometheus metrics..."
METRICS_STATUS=$(curl -sf -o /dev/null -w "%{http_code}" "$API/metrics")
echo "   /metrics status: $METRICS_STATUS"
echo ""

# 5. MLflow
echo "5. MLflow UI..."
MLFLOW_STATUS=""
for _ in 1 2 3 4 5; do
  MLFLOW_STATUS=$(curl -sf -o /dev/null -w "%{http_code}" "http://localhost:5000" || true)
  if [[ "$MLFLOW_STATUS" == "200" ]]; then
    break
  fi
  sleep 2
done
if [[ "$MLFLOW_STATUS" != "200" ]]; then
  echo "   MLflow status: unavailable"
  exit 1
fi
echo "   MLflow status: $MLFLOW_STATUS"
echo ""

echo "=== Smoke Test Complete ==="
