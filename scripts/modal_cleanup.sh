#!/bin/bash
# Kill ALL live Noeris Modal apps.
# Run this whenever you suspect zombie containers are burning credits.
#
# Usage: bash scripts/modal_cleanup.sh
#
# What it does:
#   1. Lists all ephemeral Modal apps
#   2. Stops every one of them
#   3. Confirms 0 remaining
#
# Why this is needed:
#   ModalBenchmarkSession creates ephemeral apps that persist even if
#   the Python process is killed (e.g., by killing a Claude Code agent).
#   Each live app holds an A100/H100 GPU reservation and bills container
#   uptime even with 0 function calls.

set -e

echo "Scanning for live Noeris Modal apps..."
APPS=$(modal app list 2>&1 | grep -i "ephemeral" | sed -n 's/.*\(ap-[a-zA-Z0-9]*\).*/\1/p')
COUNT=$(echo "$APPS" | grep -c "ap-" 2>/dev/null || echo 0)

if [ "$COUNT" -eq 0 ]; then
    echo "No live ephemeral apps found. All clean."
    exit 0
fi

echo "Found $COUNT live ephemeral apps. Stopping all..."
for app in $APPS; do
    modal app stop "$app" 2>/dev/null &
done
wait

sleep 2
REMAINING=$(modal app list 2>&1 | grep -ci "ephemeral" 2>/dev/null || echo 0)
echo "Done. $COUNT stopped, $REMAINING remaining."
