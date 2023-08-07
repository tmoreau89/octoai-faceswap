#!/bin/bash

set -euxo pipefail

docker_port=$(docker port faceswap 8000/tcp)
ENDPOINT=localhost:${docker_port##*:}

echo "{\"image\": \"" > request.json
base64 my_test_image.jpg >> request.json
echo "\"}" >> request.json

curl -X POST http://${ENDPOINT}/predict \
    -H "Content-Type: application/json" \
    --data @request.json | jq -r ".completion.image" | base64 -d > output.png
