#!/usr/bin/env bash
# =====================================================================
# Internal Production Deployment Plan — Wave 1 item 3 (companion)
# Backup Caddy's persistent state — the Let's Encrypt account key and
# all issued certificates / OCSP staples — to local disk.
#
# Why this matters: each Let's Encrypt account is rate-limited to
# ~50 issuances per registered domain per week. Losing caddy_data
# forces a fresh registration + reissuance, which can quietly hit the
# rate limit during a chained "rebuild + reissue" recovery scenario.
# Backing up caddy_data is cheap (~1 MB) and turns a potential
# multi-hour outage into a 2-second restore.
#
# Usage:
#     ./backup_caddy.sh                            # uses defaults
#     BACKUP_DIR=/mnt/storage/aix ./backup_caddy.sh
# =====================================================================

set -euo pipefail

# --- Configuration ---------------------------------------------------
VOLUME_NAME="${CADDY_VOLUME_NAME:-aix-caddy-data}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/aix}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

# --- Pre-flight ------------------------------------------------------
log()  { printf '%s | INFO  | %s\n'  "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
fail() { printf '%s | ERROR | %s\n'  "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >&2; exit 1; }

# Confirm the named volume exists. ``docker volume inspect`` exits 1 on
# missing volume, which we want to surface clearly.
docker volume inspect "$VOLUME_NAME" >/dev/null 2>&1 \
	|| fail "docker volume '$VOLUME_NAME' not found (is the stack up?)"

mkdir -p "$BACKUP_DIR"
[[ -w "$BACKUP_DIR" ]] || fail "backup dir '$BACKUP_DIR' is not writable by $(id -un)"

# --- Run the backup --------------------------------------------------
TIMESTAMP="$(date -u +'%Y-%m-%dT%H-%M-%SZ')"
FINAL_PATH="$BACKUP_DIR/caddy-${TIMESTAMP}.tar.gz"
PARTIAL_PATH="${FINAL_PATH}.partial"

log "snapshotting volume '$VOLUME_NAME' → ${PARTIAL_PATH}"

# Run a throwaway alpine container that bind-mounts the volume read-only
# at /data and the backup directory at /backup, then tars /data into a
# timestamped archive. Read-only on /data prevents the backup itself
# from racing against Caddy's renewal writes (Caddy still writes via the
# main container's read-write mount).
docker run --rm \
	-v "$VOLUME_NAME":/data:ro \
	-v "$BACKUP_DIR":/backup \
	alpine \
	sh -c "tar czf '/backup/$(basename "$PARTIAL_PATH")' -C /data ."

# Atomic publish: rename only after tar succeeds with non-trivial size.
size="$(stat -c '%s' "$PARTIAL_PATH" 2>/dev/null || stat -f '%z' "$PARTIAL_PATH")"
if (( size < 1024 )); then
	rm -f "$PARTIAL_PATH"
	fail "tar produced suspiciously small output ($size bytes); aborting"
fi
mv "$PARTIAL_PATH" "$FINAL_PATH"
log "✓ caddy state archived ($size bytes) → $FINAL_PATH"

# --- Retention prune -------------------------------------------------
log "pruning caddy backups older than ${BACKUP_RETENTION_DAYS} days"
deleted_count=0
while IFS= read -r -d '' old_file; do
	rm -f -- "$old_file"
	deleted_count=$((deleted_count + 1))
	log "  removed $(basename "$old_file")"
done < <(find "$BACKUP_DIR" -maxdepth 1 -type f -name 'caddy-*.tar.gz' \
                   -mtime +"$BACKUP_RETENTION_DAYS" -print0)
log "pruned ${deleted_count} expired caddy backup(s)"

log "done"
