#!/usr/bin/env bash
# Downloads the demo's models at build time, because they are too large to keep in git.
# demo/export reproduces the files. To change one, publish a new release and update its JSON in web/models.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/web/models"
for details in model.json histo.json; do
  FILE="$(node -p "require('./$details').file")"
  RELEASE="$(node -p "require('./$details').release")"
  SHA256="$(node -p "require('./$details').sha256")"
  curl -fsSL --retry 3 -o "$FILE" "https://github.com/coderback/colonscan/releases/download/$RELEASE/$FILE"
  echo "$SHA256  $FILE" | sha256sum -c -
done
