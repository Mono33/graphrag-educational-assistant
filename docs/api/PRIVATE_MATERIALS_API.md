# Private materials API

The private-material surface is mounted at `/api/v1/materials` and appears in
the existing Swagger UI at `/docs`. Every operation acts as one owner: a teacher
with their established WebUI authentication, or a chatbot platform using HTTP
Basic service credentials and acting for the end user named in
`X-End-User-Id` -- each end user is its own owner (`aix.api.principals`,
2026-10-14). Application errors carry
`{"detail": {"code", "message"}}`; standard request-validation errors retain
FastAPI's validation shape. For the chatbot flow (sessions, turns, downloads) see
[`Chatbot_API_Guide.md`](Chatbot_API_Guide.md).

## Compatibility gate

Reads are always available. Upload, version, lesson-attach and retirement
operations require:

```env
AIX_MATERIAL_LIBRARY_ENABLED=true
```

Keep it `false` until database migrations, upload/blob volume restoration and
`python scripts/ops/audit_material_storage.py` all succeed. With the flag off,
the established attachment remove-and-retire behavior is unchanged.

## Endpoints

- `GET /api/v1/materials` — list the caller's active materials.
- `POST /api/v1/materials` — multipart upload (`file`; optional
  `display_name`, `purpose`, `lesson_id`).
- `GET /api/v1/materials/{material_id}` — detail and immutable versions.
- `POST /api/v1/materials/{material_id}/versions` — append a multipart version.
- `GET /api/v1/materials/{material_id}/versions/{version_id}/source` — download
  owned source bytes.
- `POST /api/v1/materials/{material_id}/attach` — attach a ready version to an
  owned lesson. The JSON body contains `lesson_id` and optional `version_id`.
- `DELETE /api/v1/materials/{material_id}/lessons/{lesson_id}` — detach only
  from that lesson; the material remains reusable in the owner's library.
- `DELETE /api/v1/materials/{material_id}` — retire the material globally.
- `GET|POST /api/v1/materials/saved/resources` and
  `DELETE /api/v1/materials/saved/resources/{resource_id}` — manage bookmarks
  for external media separately from private source documents.

Wrong-owner IDs return `404`, including nested material/version combinations.
Uploads pass the same size, extension and magic-byte validation used by the
WebUI and queue the existing durable ingestion pipeline. API-created lesson
bindings are mirrored into the legacy lesson manifest so the attachment remains
visible in the current WebUI and usable by older rendering paths.

For the pilot, the AIxLearning integration should use its instance-specific
HTTP Basic credentials plus an opaque `X-End-User-Id`, persist the returned
UUIDs, poll the material detail until the top-level status reports `ready`, and
only then start a lesson run. A signed per-user/service identity remains the
recommendation before production because the platform and GraphRAG communicate
between separate VMs over the network.
