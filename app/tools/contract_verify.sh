#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
fixdir="tests/fixtures"

if [[ -z "${BANK_SERVICE_AUTH_SECRET:-}" ]]; then
  echo "BANK_SERVICE_AUTH_SECRET must be set" >&2
  exit 1
fi

post() {
  local name="$1"
  echo "=== $name ==="
  resp_headers=$(mktemp)
  resp_body=$(mktemp)
  request_body=$(mktemp)
  bank_user_id=$(python -c 'import uuid; print(uuid.uuid4())')
  jq --arg bankUserId "$bank_user_id" '. + {bankUserId: $bankUserId}' \
    "${fixdir}/${name}.json" > "$request_body"
  curl -s -D "$resp_headers" -H "Content-Type: application/json" \
    -H "X-Bank-Core-Auth: ${BANK_SERVICE_AUTH_SECRET}" \
    --data-binary "@$request_body" \
    "${BASE_URL}/api/v1/kyc/aggregate" > "$resp_body"

  req_id=$(grep -i '^x-request-id:' "$resp_headers" | awk -F': ' '{print $2}' | tr -d '\r')
  decision=$(jq -r '.decision' < "$resp_body")
  reasons=$(jq -r '.reasons|join(",")' < "$resp_body")
  echo "request_id: ${req_id}"
  echo "decision  : ${decision}"
  echo "reasons   : ${reasons}"
  echo
  rm -f "$resp_headers" "$resp_body" "$request_body"
}

post approve
post under_review
post reject
