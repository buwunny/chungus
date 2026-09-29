#!/bin/sh
# Renew (or revoke) the registry's delegation from the machine that holds the root key.
#
#   ./chungus-delegate.sh https://YOUR_DOMAIN            # delegate for another 90 days
#   ./chungus-delegate.sh https://YOUR_DOMAIN --revoke   # end the delegation now
#
# Root key: ~/chungus-root.key, or set CHUNGUS_ROOT_KEY. Days: 90, or set DAYS.
set -eu

REGISTRY="${1:?usage: $0 https://YOUR_DOMAIN [--revoke]}"
REGISTRY="${REGISTRY%/}"
KEY="${CHUNGUS_ROOT_KEY:-$HOME/chungus-root.key}"
DAYS="${DAYS:-90}"

command -v chungus >/dev/null || { echo "chungus is not installed" >&2; exit 1; }
[ -r "$KEY" ] || { echo "can't read the root key at $KEY" >&2; exit 1; }

if [ "${2:-}" = "--revoke" ]; then
  exec chungus delegate --revoke --key "$KEY" --registry "$REGISTRY"
fi

# The registry signs its head with the online key, so read the key from there.
head=$(curl -fsS "$REGISTRY/v1/head")
online=$(printf '%s' "$head" | grep -o '"key":"chungus1[0-9a-f]*"' | head -n1 | cut -d'"' -f4)
operator=$(printf '%s' "$head" | grep -o '"operator":"chungus1[0-9a-f]*"' | cut -d'"' -f4)

if [ -z "$online" ]; then
  echo "couldn't read the registry's head from $REGISTRY/v1/head" >&2
  exit 1
fi
if [ "$online" = "$operator" ]; then
  echo "the registry is still signing with the root key, so its online key isn't visible here." >&2
  echo "on the VPS: docker compose logs registry | grep 'online key'" >&2
  exit 1
fi

echo "online key: $online"
chungus delegate "$online" --days "$DAYS" --key "$KEY" --registry "$REGISTRY"
