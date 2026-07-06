#!/bin/bash
set -e

echo "=== boot diagnostics ${APP_VERSION:-unknown} ==="
sha256sum /app/start.sh 2>/dev/null || md5sum /app/start.sh 2>/dev/null || true

mkdir -p /data/models
for f in /opt/seed_models/*; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    dest="/data/models/$base"
    case ";${MODEL_PRESERVE:-};" in
        *";${base};"*)
            echo "preserved (MODEL_PRESERVE): $dest"
            continue
            ;;
    esac
    if [ ! -f "$dest" ]; then
        cp "$f" "$dest"
        echo "seeded model (new): $dest"
    else
        img_sha=$(sha256sum "$f" | cut -d' ' -f1)
        disk_sha=$(sha256sum "$dest" | cut -d' ' -f1)
        if [ "$img_sha" != "$disk_sha" ]; then
            cp "$f" "$dest"
            echo "updated model: $dest"
        fi
    fi
done

if [ -f /opt/seed_data/feeds.db ]; then
    current_size=0
    if [ -f /data/feeds.db ]; then
        current_size=$(stat -c%s /data/feeds.db 2>/dev/null || stat -f%z /data/feeds.db 2>/dev/null || echo 0)
    fi
    if [ "$current_size" -lt 50000 ]; then
        cp /opt/seed_data/feeds.db /data/feeds.db
        echo "seeded feeds.db"
    fi
fi

python scripts/bootstrap_db.py

python -m app.main &
BOT_PID=$!

trap 'echo "[start.sh] shutting down bot pid=$BOT_PID"; kill -TERM $BOT_PID 2>/dev/null || true; wait $BOT_PID 2>/dev/null; exit 0' TERM INT

echo "[start.sh] bot pid=$BOT_PID"

wait $BOT_PID
EXIT_CODE=$?
echo "[start.sh] bot exited code=$EXIT_CODE"
exit $EXIT_CODE
