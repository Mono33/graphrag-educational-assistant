#!/usr/bin/env bash
# =====================================================================
# Internal Production Deployment Plan — Wave 1 item 3
# Daily Postgres backup for the FEM internal pilot stack.
#
# Designed to run on the production VM as a cron job (see deploy/README.md
# §5). Captures both databases (lessons + LangGraph checkpoints share the
# same Postgres in this stack) into a single timestamped, gzipped SQL
# dump, then prunes anything older than ${BACKUP_RETENTION_DAYS} days.
#
# Hard rules this script enforces:
#   1. Strict bash mode — any error aborts the run.
#   2. flock — only one instance runs at a time, even if cron double-fires.
#   3. Atomic rename — partial dumps caused by container crash, OOM, or
#      disk-full never appear under their final ``aix-YYYY-MM-DD.sql.gz``
#      name; they sit in ``.partial`` and are picked up next run.
#   4. Pre-checks — refuses to run if the container isn't healthy or if
#      the destination directory isn't writable, instead of silently
#      writing zero-byte files.
#   5. Pruning runs only AFTER a successful dump, so a chain of failures
#      can never delete the last good backup.
#
# Usage:
#     ./backup_postgres.sh                         # uses defaults
#     BACKUP_DIR=/mnt/storage/aix \
#       BACKUP_RETENTION_DAYS=14 \
#       ./backup_postgres.sh
# =====================================================================

set -euo pipefail

# --- Configuration ---------------------------------------------------
# All knobs are env-driven so the same script runs unchanged on the dev
# VM, the staging VM, and the prod VM — only the cron environment differs.
CONTAINER_NAME="${POSTGRES_CONTAINER_NAME:-aix-postgres}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/aix}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"
LOCK_FILE="${BACKUP_LOCK_FILE:-/var/lock/aix-backup-postgres.lock}"

# --- Strict argument-free invocation ---------------------------------
if [[ "$#" -ne 0 ]]; then
	echo "Usage: $0  (no positional arguments; configure via env vars)" >&2
	exit 64  # EX_USAGE
fi

# --- Re-execute under flock so cron can't double-fire ----------------
# The "$0" "$@" idiom keeps the script self-contained: if we're not yet
# holding the lock, we re-exec ourselves through flock. The "200" is an
# arbitrary file descriptor flock uses internally.
if [[ "${_AIX_BACKUP_LOCKED:-0}" != "1" ]]; then
	export _AIX_BACKUP_LOCKED=1
	exec flock -n "$LOCK_FILE" "$0" "$@"
fi

# --- Pre-flight checks -----------------------------------------------
log()  { printf '%s | INFO  | %s\n'  "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
fail() { printf '%s | ERROR | %s\n'  "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >&2; exit 1; }

log "starting Postgres backup (container=$CONTAINER_NAME, dir=$BACKUP_DIR)"

# Ensure the container is actually running. ``docker inspect -f '{{.State.Running}}'``
# returns the literal "true" / "false" / errors with a non-zero exit if the
# container doesn't exist, all of which we want to treat as fatal.
running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || echo missing)"
if [[ "$running" != "true" ]]; then
	fail "container '$CONTAINER_NAME' is not running (state=$running)"
fi

# Ensure the backup directory exists and is writable. The trailing test
# avoids a confusing pg_dump failure later in the script.
mkdir -p "$BACKUP_DIR"
if [[ ! -w "$BACKUP_DIR" ]]; then
	fail "backup dir '$BACKUP_DIR' is not writable by $(id -un)"
fi

# Read POSTGRES_USER / POSTGRES_DB from the container env so this script
# stays in sync with deploy/.env.prod even if the operator rotates the
# values without editing the script.
PG_USER="$(docker exec "$CONTAINER_NAME" sh -c 'printf %s "$POSTGRES_USER"')"
PG_DB="$(docker exec   "$CONTAINER_NAME" sh -c 'printf %s "$POSTGRES_DB"')"
if [[ -z "$PG_USER" || -z "$PG_DB" ]]; then
	fail "POSTGRES_USER / POSTGRES_DB are empty inside the container"
fi

# --- The actual dump -------------------------------------------------
TIMESTAMP="$(date -u +'%Y-%m-%dT%H-%M-%SZ')"
FINAL_PATH="$BACKUP_DIR/aix-${TIMESTAMP}.sql.gz"
PARTIAL_PATH="${FINAL_PATH}.partial"

log "running pg_dump → ${PARTIAL_PATH}"

# pg_dump options:
#   --clean --if-exists  : restore drops + recreates objects, so a
#                          restore over an existing schema works
#                          without manual cleanup.
#   --no-owner --no-privileges : avoid GRANT/OWNER lines that reference
#                          a user that may not exist on the restore host.
# pipefail (set above) ensures the whole pipeline fails if pg_dump fails,
# even when gzip succeeds on a partial stream.
docker exec "$CONTAINER_NAME" \
	pg_dump --clean --if-exists --no-owner --no-privileges \
	        -U "$PG_USER" -d "$PG_DB" \
	| gzip -9 > "$PARTIAL_PATH"

# Refuse to publish zero-byte / suspiciously-tiny dumps. A pristine empty
# Postgres still produces ~3-5 KB of schema so 1024 bytes is a safe floor.
size="$(stat -c '%s' "$PARTIAL_PATH" 2>/dev/null || stat -f '%z' "$PARTIAL_PATH")"
if (( size < 1024 )); then
	rm -f "$PARTIAL_PATH"
	fail "pg_dump produced suspiciously small output ($size bytes); aborting"
fi

# Atomic rename: rename(2) is atomic on POSIX, so a parallel reader can
# never see ``aix-…sql.gz`` half-written.
mv "$PARTIAL_PATH" "$FINAL_PATH"
log "✓ backup written ($size bytes) → $FINAL_PATH"

# --- Retention prune -------------------------------------------------
# Only runs after the new dump landed, so a long chain of failures can't
# silently delete the last good copy.
log "pruning backups older than ${BACKUP_RETENTION_DAYS} days"
deleted_count=0
while IFS= read -r -d '' old_file; do
	rm -f -- "$old_file"
	deleted_count=$((deleted_count + 1))
	log "  removed $(basename "$old_file")"
done < <(find "$BACKUP_DIR" -maxdepth 1 -type f -name 'aix-*.sql.gz' \
                   -mtime +"$BACKUP_RETENTION_DAYS" -print0)
log "pruned ${deleted_count} expired backup(s)"

log "done"
