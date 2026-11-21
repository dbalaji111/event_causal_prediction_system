# Gold Event Risk Terminal: live server

A single web service that serves the dashboard and analyses any pasted gold headline live. The headline is embedded with
the same model as the article corpus (`all-MiniLM-L6-v2`). The service then retrieves matured historical precedents from
the full event history, runs the KDE posterior at 7, 15 and 30 days, and scores the result.

```
Dockerfile              (repo root)  image: install, download the model, build the data, run the server
.dockerignore           (repo root)  keeps the image to what the server needs
deploy/build.py         build step:  event memory, precomputed feed, KDE results, dashboard data, 3D map
deploy/server.py        FastAPI app: the page, /api/analyze, /api/health
deploy/test_server.py   tests (run after build.py; the model is replaced by stand-ins)
deploy/data/            optional real price files (see below)
```

## Deploy on Railway

1. In Railway: **New Project → Deploy from GitHub repo** → choose `event_causal_prediction_system`.
2. In the service's **Settings → Source**, set the branch to the one containing this folder.
3. Railway finds the `Dockerfile` at the repo root and builds it. The first build takes several minutes, because it
   installs PyTorch, downloads the model and builds the data. Railway sets `PORT`; the server listens on it.
4. **Settings → Networking → Generate Domain** gives you the public address.
5. Optional: **Settings → Deploy → Healthcheck Path** = `/api/health`. The server answers it once the model and event
   history are loaded, so traffic only switches to a new deploy when it is ready.

Railway's settings pages change; if a name above differs, the setting is in the service's Settings tab.

## Use real prices

Without a price file, prices come from an illustrative approximation, and the page says so in a banner. To use real data,
add these files before deploying:

- `deploy/data/gold_prices.csv`: columns `Date`, `XAUUSD`, optional `GC_M1`, `GC_M2` (daily, at least one year).
- `deploy/data/gvz.csv` (optional): columns `Date` as MM/DD/YYYY and `Price`.

The repo is public. Committing licensed price data to it publishes that data, so either make the repo private (Railway
deploys private repos) or check that your data licence allows it.

## Settings (Railway → Variables)

| Variable | Default | Meaning |
|---|---|---|
| `RATE_LIMIT_PER_MINUTE` | 20 | analyses allowed per visitor IP per minute |
| `MAX_CONCURRENT_ANALYSES` | 2 | analyses run at the same time; extra requests wait up to 30 s |
| `TORCH_THREADS` | 2 | CPU threads for the embedding model |

## API

`POST /api/analyze` with `{"headline": "...", "date": "YYYY-MM-DD"}`; `date` is optional and defaults to today.
It returns the analogues, the forecast for each horizon, the score, and a `live` block. `live.price_context` is `false`
when the price file does not cover the event date; similarity still works, but the price-shape and price-level parts of
the weighting are neutral. Interactive API docs are at `/api/docs`.

A precedent only counts once its whole 30-day outcome window had closed before the event being analysed
(`analogue_maturity_days` in `/api/health`), so no result uses returns that overlap the event itself.

## Run locally

```
docker build -t gold-terminal .
docker run -p 8000:8000 gold-terminal        # then open http://localhost:8000
```
