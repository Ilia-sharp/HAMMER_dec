#!/usr/bin/env bash
set -Eeuo pipefail

BASE_URL="$(printenv BASE_URL || printf '%s' 'http://localhost:8000')"
DEMO_IMAGE="$(printenv DEMO_IMAGE || printf '%s' 'static/img/demo.jpg')"
READY_TIMEOUT_SECONDS="$(printenv READY_TIMEOUT_SECONDS || printf '%s' '150')"
TMP_RESULT="$(mktemp)"
trap 'rm -f "$TMP_RESULT"' EXIT

say() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
fail() { printf '\nОШИБКА: %s\n' "$*" >&2; exit 1; }

say "Проверка liveness: $BASE_URL/health"
curl --silent --show-error --fail "$BASE_URL/health" >/dev/null || fail "GET /health не прошёл."

say "Ожидание готовности модели через /ready (до $READY_TIMEOUT_SECONDS секунд)"
start="$(date +%s)"
while true; do
  code="$(curl --silent --show-error --output "$TMP_RESULT" --write-out '%{http_code}' "$BASE_URL/ready" || true)"
  if [[ "$code" == "200" ]]; then break; fi
  now="$(date +%s)"
  if (( now - start >= READY_TIMEOUT_SECONDS )); then
    printf 'Последний ответ /ready: '; cat "$TMP_RESULT" 2>/dev/null || true
    fail "Модель не стала готова. Проверьте best.onnx и логи."
  fi
  sleep 3
done

say "Проверка /info"
curl --silent --show-error --fail "$BASE_URL/info" -o "$TMP_RESULT" || fail "GET /info завершился ошибкой"
python - "$TMP_RESULT" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload.get("variant") == 16 and payload.get("class") == "hammer", payload
print("OK:", payload.get("service"), "model:", payload.get("model_version"))
PY

[[ -f "$DEMO_IMAGE" ]] || fail "Не найдено изображение: $DEMO_IMAGE"
say "Отправка настоящего изображения в /detect"
curl --silent --show-error --fail --output "$TMP_RESULT" \
  -F "file=@$DEMO_IMAGE" "$BASE_URL/detect" || fail "POST /detect завершился ошибкой"
python - "$TMP_RESULT" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
required = ("request_id", "model_version", "detections", "inference_ms", "count")
assert all(key in payload for key in required), payload
assert isinstance(payload["detections"], list) and payload["count"] == len(payload["detections"]), payload
print("OK: request_id=", payload["request_id"], "detections=", payload["count"], "inference_ms=", payload["inference_ms"])
PY
say "SMOKE-ТЕСТ ПРОЙДЕН"
