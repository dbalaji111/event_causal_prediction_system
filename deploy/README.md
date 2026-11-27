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
deploy/test_live.py     live terminal tests (Refinitiv replaced by a fake source)
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

## Live terminal with Refinitiv (`/live`)

A second dashboard at `/live` shows Refinitiv gold news as it arrives, scores every headline in the background, and
lets you choose **which time series the KDE is measured on** (any instrument you have stored, such as `XAU=`,
`GCc1`, `LCOc1` or `CLc1`) and **which horizons** (1 to 90 days). Changing either recomputes the outcome
distribution, the forecast table and each precedent's outcome on that series.

```
deploy/feeds/refinitiv.py    Refinitiv / LSEG Data Library adapter: headlines and daily history
deploy/feeds/store.py        SQLite store: series values and the news feed with each headline's score
deploy/feeds/pipeline.py     pull loop shared by the collector and the server
deploy/feeds/collector.py    runs next to Refinitiv and pushes to the server
deploy/live_routes.py        /live page, /api/live/* (viewer login), /api/ingest/* (collector token)
deploy/live_dashboard.html   the live dashboard
```

**Licence first.** Refinitiv content cannot normally be shown to other people under a standard Workspace licence.
That is why `/live` requires a login and is never public. Check your licence before giving anyone else access.

### Where the data comes from

Most university access is **Workspace on a desktop**, which only works on that computer. So the usual setup is:

1. On the computer with Workspace: `pip install -r deploy/feeds/requirements-collector.txt`.
2. In Workspace, open **App Key Generator** and create a key.
3. Test the connection without sending anything:
   `REFINITIV_APP_KEY=... python deploy/feeds/collector.py --check --rics "XAU=,LCOc1"`
4. Run it (keep it running while Workspace is open):
   ```
   REFINITIV_APP_KEY=... INGEST_TOKEN=... python deploy/feeds/collector.py \
       --server https://terminal.brahmanda.com.au \
       --rics "XAU=,GCc1,LCOc1,CLc1" --names "XAU=:Gold spot,GCc1:Gold future,LCOc1:Brent,CLc1:WTI" \
       --query "Topic:GOL AND Language:LEN"
   ```
   The first run sends each series' history from 1999. After that it checks news every minute and prices every hour,
   and remembers where it got to in `collector_state.json`.

If your licence includes **Data Platform** access (a cloud login), the server can pull by itself instead: set
`SERVER_REFINITIV=1`, `REFINITIV_MODE=platform`, `REFINITIV_APP_KEY`, `REFINITIV_USERNAME`, `REFINITIV_PASSWORD`,
`SERIES_RICS` and `NEWS_QUERIES`, and `INSTALL_REFINITIV=1` so the image includes the library.

The news query and the instrument codes (RICs) above are starting points: confirm them in Workspace for your entitlements.

### Railway variables for the live terminal

| Variable | Example | Meaning |
|---|---|---|
| `LIVE_USERNAME`, `LIVE_PASSWORD` | your choice | login for `/live`; without them `/live` stays off |
| `INGEST_TOKEN` | a long random string | the collector's key; the same value goes in the collector's `INGEST_TOKEN` |
| `DEFAULT_SERIES` | `XAU=` | series used to score the feed; default is the built-in gold series |
| `DATA_DIR` | `/data` | where the feed and series are stored. Add a Railway **Volume** mounted at `/data`, or the data is lost on each redeploy. If the volume is not writable, also set `RAILWAY_RUN_UID=0` |

Server-side Refinitiv only: `SERVER_REFINITIV`, `REFINITIV_MODE`, `REFINITIV_APP_KEY`, `REFINITIV_USERNAME`,
`REFINITIV_PASSWORD`, `NEWS_QUERIES` (several separated by `|`), `SERIES_RICS` (comma-separated),
`NEWS_POLL_SECONDS` (60), `SERIES_POLL_SECONDS` (3600), `INSTALL_REFINITIV=1`.

### How the KDE uses the chosen series

The precedents are found the same way whatever series you choose: by the meaning of the headline in the gold event
memory. Each precedent's path and outcome are then measured on the chosen series, and the KDE is built from those
outcomes. A precedent only counts once its longest outcome window has closed before the event, and an event whose
own window has not finished shows no realised outcome.

## Run locally

```
docker build -t gold-terminal .
docker run -p 8000:8000 gold-terminal        # then open http://localhost:8000
```
