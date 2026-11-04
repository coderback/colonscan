#!/usr/bin/env bash
# Downloads the demo's model at build time, because it is too large to keep in git.
# demo/export reproduces the file. To change it, publish a new release and update TAG and web/models/model.json.
set -euo pipefail
TAG=demo-model-e69f783a
cd "$(dirname "${BASH_SOURCE[0]}")/web/models"
FILE="$(node -p 'require("./model.json").file')"
SHA256="$(node -p 'require("./model.json").sha256')"
curl -fsSL --retry 3 -o "$FILE" "https://github.com/coderback/colonscan/releases/download/$TAG/$FILE"
echo "$SHA256  $FILE" | sha256sum -c -
