IMAGE_TAG ?= faceswap
CONTAINER_NAME ?= ${IMAGE_TAG:%:latest=%}
INFERENCE_SERVER_PORT ?= 8000

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

build:
	docker build -t "${IMAGE_TAG}" --progress string .

run:
	docker run -d --rm -p ${INFERENCE_SERVER_PORT}:8000 \
		--gpus all \
		--name "${CONTAINER_NAME}" "${IMAGE_TAG}"

bash:
	docker exec -it ${CONTAINER_NAME} bash

stop:
	docker stop "${CONTAINER_NAME}"

.PHONY: build run stop
