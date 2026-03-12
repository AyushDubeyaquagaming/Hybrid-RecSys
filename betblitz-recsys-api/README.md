# BetBlitz Recommendation API

FastAPI service that loads trained LightFM artifacts and serves real-time game recommendations.

## Setup

### 1. Export artifacts from the notebook

After running all cells in `recsys_real_data.ipynb`, export artifacts:

```bash
# From the notebook kernel
%run scripts/export_artifacts.py
```

Or copy `scripts/export_artifacts.py` contents into a notebook cell and run it. This produces the `artifacts/` directory with all required files.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The notebook uses a local LightFM build. If running outside the notebook venv, ensure the same LightFM version is installed to guarantee artifact compatibility.

### 3. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Run with Docker

```bash
sudo docker build -t betblitz-recsys-api .
sudo docker run --rm -p 8000:8000 --name betblitz-api betblitz-recsys-api
```

Validated local flow:

1. Open a terminal in `/home/ayush/lightfm_project/betblitz-recsys-api`
2. Build the image:

```bash
sudo docker build -t betblitz-recsys-api .
```

3. Start the API container:

```bash
sudo docker run --rm -p 8000:8000 --name betblitz-api betblitz-recsys-api
```

4. In a second terminal, test the containerized API:

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id":"5097103780","top_k":5}'
```

5. Optional: open Swagger UI at `http://localhost:8000/docs`

Notes:

- Use `sudo` if your user does not have permission to access `/var/run/docker.sock`.
- Rebuild the image whenever `app/`, `requirements.txt`, `Dockerfile`, or exported files under `artifacts/` change.
- If nothing changed since the last successful build, you can skip the build step and only run the container.

## API Usage

### Health check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v0.1-real-data",
  "n_users": 4321,
  "n_items": 150
}
```

### Recommendations for a known user

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "1244444444", "top_k": 5}'
```

### Cold-start user (not in training data)

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "new_user_999"}'
```

Returns popularity-based fallback with `"is_cold_start": true`.

### Include already-played games

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "1244444444", "top_k": 5, "exclude_played": false}'
```

## Running tests

```bash
pytest tests/
```

## Environment variables

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
|---|---|---|
| `ARTIFACT_DIR` | `./artifacts` | Path to model artifacts |
| `DEFAULT_TOP_K` | `5` | Default number of recommendations |
| `MAX_TOP_K` | `20` | Maximum allowed top_k |
| `PREDICT_NUM_THREADS` | `1` | LightFM predict threads |
