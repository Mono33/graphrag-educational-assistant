#!/usr/bin/env bash
# =====================================================================
# Internal Production Deployment Plan — Wave 1 item 3
# Restore a Postgres backup produced by ``backup_postgres.sh`` into the
# running ``aix-postgres`` container.
#
# DESTRUCTIVE — this overwrites the current database. The script refuses
# to run unless the operator passes ``--yes`` (or sets AIX_RESTORE_YES=1
# for cron-style automated restores on a fresh side-VM).
#
# Pairs with backup_postgres.sh: dumps are written with ``--clean
# --if-exists`` so a plain ``psql`` restore drops every existing object
# before recreating it. We do NOT need to drop+recreate the database
# itself, which keeps the connection pool / role settings intact.
#
# Usage:
#     ./restore_postgres.sh /var/backups/aix/aix-2026-05-16T03-00-00Z.sql.gz --yes
#     AIX_RESTORE_YES=1 ./restore_postgres.sh /path/to/aix-…sql.gz
# =====================================================================

set -euo pipefail

# --- Configuration ---------------------------------------------------
CONTAINER_NAME="${POSTGRES_CONTAINER_NAME:-aix-postgres}"

# --- Argument parsing ------------------------------------------------
BACKUP_FILE=""
CONFIRMED="${AIX_RESTORE_YES:-0}"

usage() {
	cat <<-EOF >&2
		Usage: $0 <backup-file.sql.gz> [--yes]

		DESTRUCTIVE. Restores the named gzipped pg_dump into the
		running '${CONTAINER_NAME}' container, overwriting current data.

		Pass --yes to skip the interactive confirmation, or set
		AIX_RESTORE_YES=1 in the environment.
	EOF
	exit 64  # EX_USAGE
}

while (( "$#" )); do
	case "$1" in
		-h|--help)  usage ;;
		--yes)      CONFIRMED=1; shift ;;
		--)         shift; BACKUP_FILE="${1:-}"; break ;;
		-*)         echo "unknown flag: $1" >&2; usage ;;
		*)          if [[ -z "$BACKUP_FILE" ]]; then BACKUP_FILE="$1"; else
		            echo "unexpected argument: $1" >&2; usage; fi
		            shift ;;
	esac
done

[[ -n "$BACKUP_FILE" ]] || usage

# --- Pre-flight checks -----------------------------------------------
log()  { printf '%s | INFO  | %s\n'  "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
fail() { printf '%s | ERROR | %s\n'  "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >&2; exit 1; }

[[ -f "$BACKUP_FILE" ]] || fail "backup file not found: $BACKUP_FILE"
[[ -r "$BACKUP_FILE" ]] || fail "backup file not readable: $BACKUP_FILE"

# Sanity: the file should be gzip-compressed. ``file -b`` returns the MIME
# type / human description; ``gunzip -t`` actually verifies the integrity.
if ! gunzip -t "$BACKUP_FILE" 2>/dev/null; then
	fail "backup file appears corrupted (gunzip -t failed): $BACKUP_FILE"
fi

# Ensure the container is running before we even prompt the operator.
running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || echo missing)"
if [[ "$running" != "true" ]]; then
	fail "container '$CONTAINER_NAME' is not running (state=$running)"
fi

# Resolve credentials live from the container, mirroring backup_postgres.sh.
PG_USER="$(docker exec "$CONTAINER_NAME" sh -c 'printf %s "$POSTGRES_USER"')"
PG_DB="$(docker exec   "$CONTAINER_NAME" sh -c 'printf %s "$POSTGRES_DB"')"
[[ -n "$PG_USER" && -n "$PG_DB" ]] || fail "POSTGRES_USER / POSTGRES_DB empty inside container"

# --- Confirmation ----------------------------------------------------
if [[ "$CONFIRMED" != "1" ]]; then
	echo
	echo "  About to restore over the LIVE database '$PG_DB' in container '$CONTAINER_NAME'."
	echo "  Backup source:  $BACKUP_FILE"
	echo "  This will DROP every existing object before recreating it."
	echo
	read -r -p "  Type 'yes' to proceed: " answer
	if [[ "$answer" != "yes" ]]; then
		echo "  aborted." >&2
		exit 1
	fi
fi

# --- Take a safety dump first ----------------------------------------
# Tiny insurance: if the restore corrupts the schema, the operator still
# has the prior state available locally. We don't prune this one.
SAFETY_DIR="${BACKUP_DIR:-/var/backups/aix}"
mkdir -p "$SAFETY_DIR"
SAFETY_FILE="${SAFETY_DIR}/pre-restore-$(date -u +'%Y-%m-%dT%H-%M-%SZ').sql.gz"
log "saving safety snapshot of current DB → $SAFETY_FILE"
docker exec "$CONTAINER_NAME" \
	pg_dump --clean --if-exists --no-owner --no-privileges \
	        -U "$PG_USER" -d "$PG_DB" \
	| gzip -9 > "$SAFETY_FILE"

# --- Restore ---------------------------------------------------------
# psql -v ON_ERROR_STOP=1 makes the whole restore abort on the first error
# instead of plowing through and leaving a corrupted DB. The ``--clean
# --if-exists`` lines in the dump (added by backup_postgres.sh) handle
# the drop-then-create pattern cleanly.
log "restoring $(basename "$BACKUP_FILE") into '$PG_DB'"
gunzip -c "$BACKUP_FILE" \
	| docker exec -i "$CONTAINER_NAME" \
	    psql -v ON_ERROR_STOP=1 -U "$PG_USER" -d "$PG_DB"

log "✓ restore complete (safety snapshot kept at $SAFETY_FILE)"
