#!/bin/sh

floyd run --gpu --tensorboard \
	--data davidmack/datasets/english2cypher:/data \
	--env tensorflow-1.8 \
	"python -m e2c.train --input-dir /data --output-dir /output --model-dir /output/model --max-steps 400 --vocab-size 120"