#!/usr/bin/env bash
#
# Uploads every current file in this folder to https://github.com/baransrk0/Odine-demo
# directly on the `main` branch.
#
# - Respects .gitignore, so your .env (secrets) is NOT uploaded.
# - Commits whatever is currently in the folder and pushes it to main.
#
# Run it from the project root:  bash push_to_github.sh

set -euo pipefail

REMOTE_URL="https://github.com/baransrk0/Odine-demo.git"
TARGET_BRANCH="main"
COMMIT_MSG="Initial import: Odine demo"

cd "$(dirname "$0")"

# Point 'origin' at the repo (add or update).
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

# Stage everything, drop macOS cruft, and commit.
git add -A
git ls-files --cached | grep -i 'DS_Store' | xargs -r git rm --cached -q || true
git commit -m "$COMMIT_MSG" || echo "Nothing new to commit — pushing existing state."

# Push current state straight to main. --force so it works even if the repo isn't empty.
git push -u --force origin HEAD:"$TARGET_BRANCH"

echo
echo "Done. All files are now at https://github.com/baransrk0/Odine-demo (branch: $TARGET_BRANCH)."
