# Production Deploy — Internal FEM Pilot (Wave 1)

This folder contains everything needed to stand up the Agentic GraphRAG
stack on a fresh Ubuntu VM and serve it under
`https://agente.aiforlearning.digital`.

```text
deploy/
├── docker-compose.prod.yml   # 3-service stack: app + postgres + caddy
├── Caddyfile                 # Reverse proxy + auto-HTTPS
├── .env.prod.example         # Production env template (no secrets)
├── scripts/                  # Backup / restore helpers
├── .gitignore                # Excludes the real .env.prod
└── README.md                 # This file
```

The build context is the **repo root** (`..`), so the existing top-level
`Dockerfile` is reused. Production images install from the generated
`requirements.lock.txt` lockfile, run as the non-root `aix` user, listen
on container port `8765`, and expose a baked-in `/api/v1/health`
`HEALTHCHECK`. Nothing here duplicates the dev compose file at the repo
root — that one stays in place for Angelo's local build/push workflow.

> **2026-07-02 — production image hardening.** The top-level `Dockerfile` was
> corrected to (a) serve on **8765** (it previously started uvicorn on port 80,
> which mismatched Caddy's `reverse_proxy app:8765` and the compose healthcheck →
> would 502), (b) install from `requirements.lock.txt` (hash-pinned), (c) run as
> the non-root `aix` user, and (d) bake in the `/api/v1/health` HEALTHCHECK. If
> you built an image before this date, rebuild.

---

## 0. Deploy flow (branches + CD) and pre-flight checklist — READ FIRST

### 0.1 Branch → deploy flow

The FEM production VM deploys from the **`production`** branch on the **`fem`**
remote (`fem/production`) via FEM's GitHub continuous-deployment pipeline. To
ship the current production-ready code:

```bash
git checkout production
git merge --no-ff chore/repo-reorg      # bring the reorg + all Phase A/B work in
git push fem production                  # this push is what triggers FEM's CD
```

> ⚠️ **Unverified assumption — confirm with FEM before Friday.** The above only
> deploys the stack **if FEM's CD pipeline actually builds/runs
> `deploy/docker-compose.prod.yml`**. If their pipeline has its own compose/config
> and doesn't know about this folder, pushing to `production` will *not* stand up
> this stack. Until that's confirmed, prefer the **manual first deploy** (§3) over
> "push and pray".

### 0.2 Pre-flight checklist (tick before the first production deploy)

- [ ] **App image** rebuilt after the 2026-07-02 Dockerfile fix (serves 8765).
- [ ] **`deploy/.env.prod`** created on the VM from `.env.prod.example`,
      `chmod 600`, and every `replace-with-...` filled — at minimum:
  - [ ] `POSTGRES_PASSWORD` (strong, random)
  - [ ] `NEO4J_PASSWORD` (prod Neo4j)
  - [ ] `OPENROUTER_API_KEY` (production-billing key, not a personal one)
  - [ ] `WEBUI_AUTH_SECRET` — **generate a fresh one on the VM**:
        `python -c "import secrets; print(secrets.token_urlsafe(48))"`
        (never reuse a value that has been pasted into git/chat/logs)
  - [ ] `AIX_TLS_EMAIL` — a **real, monitored** mailbox (Let's Encrypt needs it)
  - [ ] `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (optional but recommended —
        this is the observability goal), `SENTRY_DSN` (optional)
- [ ] **DNS**: `agente.aiforlearning.digital` → VM IP (`91.99.147.27`) — done.
- [ ] **Ports 80/443 free** on the reused VM for Caddy. The existing GraphRAG
      instance must **not** already bind them with another reverse proxy, or
      Caddy can't start / can't obtain the TLS cert. Verify: `sudo ss -tlnp | grep -E ':80 |:443 '`.
- [ ] **Neo4j reachable** from the VM: `bolt+s://graph.aiforlearning.digital:7687`.
- [ ] **First deploy validated**: build + up + `GET /api/v1/health` returns `200`
      (this is also the still-open "production image build + healthcheck on Linux"
      gate in the deployment plan §10.4 — the port bug above is exactly what it
      would have caught).

### 0.3 Re-deploy note — artifacts volume ownership (non-root app)

The app now runs as the non-root `aix` user (uid `10001`). A **fresh**
`aix-app-artifacts` volume inherits writable ownership automatically on first
mount. Only if you are **re-deploying over a volume created by an older
root-based image** do you need to fix ownership once:

```bash
docker run --rm -v aix-app-artifacts:/a alpine chown -R 10001:10001 /a
```

---

## 1. What this stack runs

| Service    | Image                  | Public? | Purpose |
|------------|------------------------|---------|---------|
| `app`      | built from `Dockerfile`| no      | FastAPI process — `/api/v1/*` JSON+SSE + `/webui/*` HTML+SSE |
| `postgres` | `postgres:16-alpine`   | no      | Single instance, backs both `WEBUI_DATABASE_URL` and `LANGGRAPH_DATABASE_URL` |
| `caddy`    | `caddy:2-alpine`       | yes     | TLS termination + reverse proxy on `80`/`443` |

Neo4j is **not** in the compose — production talks to the existing
external instance at `${NEO4J_URI}` (`graph.aiforlearning.digital`).

---

## 2. Prerequisites on the VM

- Ubuntu 24.04 LTS (or any distro with Docker Engine 25+ and Compose v2).
- Docker Engine + Compose plugin installed:
  ```bash
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER  # log out + back in
  ```
- Firewall: only `22` / `80` / `443` open. The Postgres and app ports
  are intentionally not published.
- DNS `A` record `agente.aiforlearning.digital -> <VM_IP>` already
  pointing at the VM **before** first start (Caddy needs it to issue
  the Let's Encrypt cert).
- A real mailbox someone monitors (used in `AIX_TLS_EMAIL` for
  Let's Encrypt renewal warnings).

---

## 3. First deploy

```bash
# On the VM, in the repo root after a fresh clone:
cd deploy/
cp .env.prod.example .env.prod
chmod 600 .env.prod                 # only the deploying user can read

# Edit .env.prod and replace every "replace-with-..." placeholder.
# At minimum you must set:
#   AIX_DOMAIN, AIX_TLS_EMAIL,
#   POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB,
#   NEO4J_PASSWORD, OPENROUTER_API_KEY.
nano .env.prod

# Build the app image (pinned to the current git SHA for traceability):
GIT_SHA=$(git rev-parse --short HEAD) \
  docker compose -f docker-compose.prod.yml --env-file .env.prod build

# Start the stack in the background:
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d

# Watch the bring-up. Caddy will request a cert on the first HTTPS
# connection — confirm via: `docker compose ... logs caddy`
docker compose -f docker-compose.prod.yml --env-file .env.prod ps
docker compose -f docker-compose.prod.yml --env-file .env.prod logs -f
```

Expected health-check sequence (visible in `docker compose ps`):

1. `postgres` reaches `healthy` after ~10-20 s.
2. `app` reaches `healthy` after ~30-60 s (it pre-warms the LangGraph
   checkpointer and probes Neo4j on first request to `/api/v1/health`).
3. `caddy` is `running` immediately, but the cert appears in
   `caddy_data` only after the first HTTPS request.

Browse to `https://agente.aiforlearning.digital/api/v1/health` from a
machine outside the VM. A `200` response with
`{"status":"healthy",...}` means the stack is live end-to-end.

---

## 4. Upgrade flow

```bash
cd deploy/
git pull                # pull the new commits
GIT_SHA=$(git rev-parse --short HEAD) \
  docker compose -f docker-compose.prod.yml --env-file .env.prod build app
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d app
```

Only the `app` service rebuilds; postgres + caddy stay running. The
new container takes over within ~10 s once its health check passes.

If the new image fails its health check, compose keeps the old
container running (Docker default behaviour). Roll back by re-checking
out the previous SHA and rebuilding (see §6).

---

## 5. Backups

Postgres data lives in the named volume `aix-pg-data`; Caddy's
Let's Encrypt account + issued certs live in `aix-caddy-data`. Both
are backed by the scripts under `deploy/scripts/`:

| Script | Purpose | Default destination | Default retention |
|---|---|---|---|
| `backup_postgres.sh` | atomic gzipped `pg_dump` of the live DB | `/var/backups/aix/aix-…sql.gz` | 7 days |
| `backup_caddy.sh`    | tarball snapshot of `aix-caddy-data`     | `/var/backups/aix/caddy-…tar.gz` | 30 days |
| `restore_postgres.sh`| restore one Postgres dump (interactive)  | overwrites the live DB | — |

All three are env-tunable: set `BACKUP_DIR`, `BACKUP_RETENTION_DAYS`,
`POSTGRES_CONTAINER_NAME`, or `CADDY_VOLUME_NAME` in the cron environment
to override the defaults. They are written for Linux (the production VM);
running them from Windows local dev is not supported.

### 5.1 First-time setup on the VM

```bash
# Make the scripts executable (only required after a fresh git clone)
chmod +x deploy/scripts/*.sh

# Sanity-check Postgres backup against the running stack
sudo BACKUP_DIR=/var/backups/aix \
    deploy/scripts/backup_postgres.sh
ls -lh /var/backups/aix/
```

Expected output:

```text
2026-05-16T03:00:00Z | INFO  | starting Postgres backup ...
2026-05-16T03:00:01Z | INFO  | ✓ backup written (12345 bytes) → /var/backups/aix/aix-2026-05-16T03-00-00Z.sql.gz
2026-05-16T03:00:01Z | INFO  | pruning backups older than 7 days
2026-05-16T03:00:01Z | INFO  | done
```

### 5.2 Schedule via cron

Add to `/etc/cron.d/aix-backups` on the VM (one file, root-owned, mode
`0644`). Adjust the absolute path if the repo lives elsewhere:

```cron
# m  h  dom  mon  dow  user  command
  0  3  *    *    *    root  cd /opt/graphaixlearning && BACKUP_DIR=/var/backups/aix deploy/scripts/backup_postgres.sh >> /var/log/aix-backup.log 2>&1
 30  3  *    *    0    root  cd /opt/graphaixlearning && BACKUP_DIR=/var/backups/aix deploy/scripts/backup_caddy.sh    >> /var/log/aix-backup.log 2>&1
```

The Postgres backup runs daily at 03:00 UTC; the Caddy snapshot runs
weekly on Sundays at 03:30 UTC (cert renewals are infrequent so a
weekly tarball is plenty).

### 5.3 Restore on a fresh VM

```bash
# 1) Bring up the stack first so the postgres container exists
docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env.prod up -d postgres

# 2) Run the restore script (interactive — confirms before overwriting)
deploy/scripts/restore_postgres.sh /var/backups/aix/aix-2026-05-16T03-00-00Z.sql.gz

# 3) Bring the rest of the stack up
docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env.prod up -d
```

The script automatically takes a `pre-restore-…sql.gz` safety snapshot
before applying the named backup, so even an accidental wrong-file
selection can be undone.

For automated restore (e.g. on a side-VM during a monthly restore
drill) pass `--yes` or set `AIX_RESTORE_YES=1`.

### 5.4 Off-host copy (recommended)

Local backups protect against application-level corruption but **not**
against losing the VM. Anything that can `rsync` over SSH works; one
common pattern is a nightly job from a separate machine:

```bash
# On the off-host machine, after Postgres + Caddy backups land on the VM
rsync -avz --delete \
  fem-pilot.aix:/var/backups/aix/ \
  /mnt/storage/aix-backups/
```

This is intentionally NOT in the script — pick whichever destination
fits FEM's existing infrastructure (object storage, NAS, second VM).

---

## 6. Rollback

```bash
cd deploy/
git checkout <previous-sha>
GIT_SHA=<previous-sha> \
  docker compose -f docker-compose.prod.yml --env-file .env.prod build app
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d app
```

For a Postgres rollback, restore the most recent `aix-YYYY-MM-DD.sql.gz`
into a freshly recreated `aix-pg-data` volume. The LangGraph
checkpointer tables (`checkpoints`, `checkpoint_blobs`,
`checkpoint_writes`) and the WebUI tables are all in the same dump,
so a single restore is enough.

---

## 7. Inspecting a running stack

```bash
# Status of all services
docker compose -f docker-compose.prod.yml --env-file .env.prod ps

# Live logs (Ctrl-C to detach)
docker compose -f docker-compose.prod.yml --env-file .env.prod logs -f app
docker compose -f docker-compose.prod.yml --env-file .env.prod logs -f caddy
docker compose -f docker-compose.prod.yml --env-file .env.prod logs -f postgres

# Open a psql shell against the running Postgres
docker exec -it aix-postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"
# Inside psql, sanity-check that the LangGraph saver bootstrapped:
#   \dt
#   SELECT count(*) FROM checkpoints;

# Open a shell inside the app container (debug only)
docker exec -it aix-app /bin/bash
```

---

## 8. What is NOT yet wired (subsequent waves)

This stack closes Wave 1 items 2-4 locally in
`docs/product/Internal_Production_Deployment_Plan.md`: item 2 provides the
production compose stack, item 3 provides the backup/restore scripts plus
cron template, and item 4 provides the locked/hardened production image.
The remaining Wave 1 items are tracked there:

- **Item 3 follow-up on real infrastructure** — off-host copy destination
  plus one side-VM restore drill once the VM exists.
- **Item 4 follow-up on real infrastructure** — run a full image build
  and app healthcheck on the Linux VM before first pilot traffic.
- **Item 5** — VM provisioning runbook (Ubuntu 24.04 + Docker install
  + firewall rules) for the FEM ops handover.

Subsequent waves cover DNS + TLS lockdown (Wave 2), user auth +
guardrails + rate limiting (Wave 3), observability (Wave 4),
end-to-end smoke (Wave 5), and stakeholder rollout (Wave 6).
