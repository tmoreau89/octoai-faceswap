#!/bin/bash

set -euxo pipefail

docker_port=$(docker port faceswap 8080/tcp)
ENDPOINT=localhost:${docker_port##*:}

echo "{\"src_image\": \"" > request.json
base64 two_source.jpeg >> request.json
echo "\", \"dst_image\": \"" >> request.json
base64 two_destination.png >> request.json
echo "\"}" >> request.json

curl -X POST http://${ENDPOINT}/predict \
    -H "Content-Type: application/json" \
    --data @request.json | jq -r ".completion.image" | base64 -d > two_output.png

echo "{\"src_image\": \"" > request.json
base64 five_source.jpg >> request.json
echo "\", \"dst_image\": \"" >> request.json
base64 five_destination.png >> request.json
echo "\"}" >> request.json

curl -X POST http://${ENDPOINT}/predict \
    -H "Content-Type: application/json" \
    --data @request.json | jq -r ".completion.image" | base64 -d > five_output.png

echo "{\"src_image\": \"" > request.json
base64 five_source.jpg >> request.json
echo "\", \"dst_image\": \"" >> request.json
base64 five_destination_4.png >> request.json
echo "\"}" >> request.json

curl -X POST http://${ENDPOINT}/predict \
    -H "Content-Type: application/json" \
    --data @request.json | jq -r ".completion.image" | base64 -d > five_output_4.png

echo "{\"src_image\": \"" > request.json
base64 five_source.jpg >> request.json
echo "\", \"dst_image\": \"" >> request.json
base64 five_destination_6.png >> request.json
echo "\"}" >> request.json

curl -X POST http://${ENDPOINT}/predict \
    -H "Content-Type: application/json" \
    --data @request.json | jq -r ".completion.image" | base64 -d > five_output_6.png
