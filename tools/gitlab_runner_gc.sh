#!/usr/bin/env bash
#
# Garbage collect GitLab runner Docker volumes on a CI node.
#
# The docker executor keeps two persistent volumes per (runner, concurrent
# slot, protected/unprotected) tuple:
#
#   runner-<id>-cache-c33bcaa1fd2c77edfc3893b41966cea8   -> /builds  (build dir)
#   runner-<id>-cache-3c3f060a0374fc8bc39395164f415a70   -> /cache   (cache.zip)
#
# Neither is ever cleaned by the runner itself. The build volumes accumulate
# whatever a job leaves behind, and the cache volumes accumulate one cache.zip
# per cache key that has ever been used -- including keys from jobs that have
# since been renamed or deleted.
#
# The .gitlab-ci.yml after_script keeps the build volumes bounded going
# forward. This script handles what that cannot reach: debris written before
# after_script existed, volumes belonging to runners that have been
# de-registered, and cache keys that no longer correspond to any job.
#
# Runs its filesystem work inside a container, so docker group membership is
# enough -- no root needed. Dry run by default; pass --apply to delete.
#
# Usage:
#   tools/gitlab_runner_gc.sh                      # report only
#   tools/gitlab_runner_gc.sh --apply              # sweep debris + stale cache
#   tools/gitlab_runner_gc.sh --apply --max-age 30 # also drop volumes idle 30d+
#
set -euo pipefail

APPLY=0
CACHE_MAX_AGE_DAYS=30
VOLUME_MAX_AGE_DAYS=0   # 0 disables whole-volume removal

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)      APPLY=1; shift ;;
    --cache-age)  CACHE_MAX_AGE_DAYS="$2"; shift 2 ;;
    --max-age)    VOLUME_MAX_AGE_DAYS="$2"; shift 2 ;;
    -h|--help)    usage 0 ;;
    *)            echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

BUILDS_HASH=c33bcaa1fd2c77edfc3893b41966cea8   # md5 of "/builds"
CACHE_HASH=3c3f060a0374fc8bc39395164f415a70    # md5 of "/cache"

# Not checked with -d: the docker root is typically root-only, so an
# unprivileged member of the docker group cannot stat it even though the
# daemon can bind-mount it into the helper container just fine.
VOLUME_ROOT=$(docker info --format '{{.DockerRootDir}}')/volumes
[[ -n "$VOLUME_ROOT" ]] || { echo "cannot determine docker root dir" >&2; exit 1; }

# Any small image with a shell works; the runner helper is always present.
HELPER=${GC_HELPER_IMAGE:-$(docker images --format '{{.Repository}}:{{.Tag}}' \
          | grep -m1 gitlab-runner-helper || true)}
[[ -n "$HELPER" ]] || { echo "set GC_HELPER_IMAGE to an image with a shell" >&2; exit 1; }

echo "docker volume root : $VOLUME_ROOT"
echo "helper image       : $HELPER"
echo "mode               : $([[ $APPLY -eq 1 ]] && echo APPLY || echo 'DRY RUN (use --apply)')"
echo

in_helper() {
  docker run --rm -v "$VOLUME_ROOT":/vols "$HELPER" sh -c "$1"
}

###
### 1. Build-volume debris: core dumps and kept test scratch trees.
###
### These are pure waste -- core dumps from tests that crash on purpose, and
### the tmp-build tree that test.py --keep preserves. Removing them does not
### cost a rebuild, because the runner re-runs git clean on the next job
### anyway. This is the bulk of the reclaimable space.
###
echo "=== build volume debris ==="
FIND_DEBRIS='find /vols/*'"$BUILDS_HASH"'*/_data \( -type f -name "core.*.*.*" -o -type d -name tmp-build \) -prune 2>/dev/null'
in_helper "$FIND_DEBRIS"' -exec du -cms {} + 2>/dev/null | tail -1' || true
if [[ $APPLY -eq 1 ]]; then
  in_helper "$FIND_DEBRIS"' -exec rm -rf {} + 2>/dev/null; echo "  removed"' || true
else
  in_helper "$FIND_DEBRIS | head -10" || true
  echo "  (dry run; showing first 10 matches)"
fi
echo

###
### 2. Stale cache archives.
###
### One cache.zip per cache key, forever. A key stops being written the moment
### its job is renamed or dropped from .gitlab-ci.yml, but the archive stays.
### Anything untouched for a month is not being pulled by any current job.
###
echo "=== cache archives older than ${CACHE_MAX_AGE_DAYS}d ==="
STALE="find /vols/*${CACHE_HASH}*/_data -type f -name 'cache.zip' -mtime +${CACHE_MAX_AGE_DAYS} 2>/dev/null"
in_helper "$STALE"' -exec du -cms {} + 2>/dev/null | tail -1' || true
in_helper "$STALE"' | sed "s|.*/legion/||; s|/cache.zip||" | sort | head -20' || true
if [[ $APPLY -eq 1 ]]; then
  # Drop the metadata.json alongside each archive so the entry disappears cleanly.
  in_helper "$STALE"' | while read -r z; do rm -f "$z" "$(dirname "$z")/metadata.json"; rmdir "$(dirname "$z")" 2>/dev/null; done; echo "  removed"' || true
fi
echo

###
### 3. Volumes belonging to runners that no longer exist.
###
### Re-registering a runner mints a new token, and every volume keyed to the
### old token is orphaned permanently. Docker refuses to remove a volume that
### is attached to a container, so a running job cannot be disturbed.
###
if [[ "$VOLUME_MAX_AGE_DAYS" -gt 0 ]]; then
  echo "=== volumes with no writes in ${VOLUME_MAX_AGE_DAYS}d ==="
  mapfile -t IDLE < <(in_helper "for d in /vols/*-cache-*/_data; do
      v=\$(echo \"\$d\" | sed 's|/vols/||; s|/_data||')
      if [ -z \"\$(find \"\$d\" -newermt '-${VOLUME_MAX_AGE_DAYS} days' -maxdepth 3 2>/dev/null | head -1)\" ]; then
        echo \"\$v\"
      fi
    done" | tr -d '\r')
  if [[ ${#IDLE[@]} -eq 0 ]]; then
    echo "  none"
  else
    for v in "${IDLE[@]}"; do
      [[ -n "$v" ]] || continue
      echo "  $v"
      # docker refuses to remove a volume attached to a container, so a
      # running job cannot be disturbed by this.
      [[ $APPLY -eq 1 ]] && { docker volume rm "$v" >/dev/null 2>&1 \
        && echo "    removed" || echo "    in use, skipped"; }
    done
  fi
  echo
fi

###
### 4. Images and stopped containers.
###
echo "=== reclaimable images / containers ==="
if [[ $APPLY -eq 1 ]]; then
  docker container prune -f
  docker image prune -f
else
  docker system df
fi
echo

echo "=== resulting usage ==="
docker system df
# df from inside the helper, since the docker root is usually root-only.
in_helper 'df -h /vols | tail -n +1' || true
