#!/bin/bash

MESSAGE="${1:-Auto-commit: $(date '+%Y-%m-%d %H:%M:%S')}"
PUSH="${2:-true}"

echo "Staging all changes..."
git add -A

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Changes detected. Committing..."
    git commit -m "$MESSAGE"
    
    if [[ "$PUSH" != "nopush" ]]; then
        echo "Pushing to remote..."
        git push
        echo "Successfully pushed to GitHub!"
    fi
else
    echo "No changes to commit."
fi

git status --short