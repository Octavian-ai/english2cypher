#!/bin/sh

floyd run --gpu --tensorboard \
	--data davidmack/datasets/english2cypher/3:/data \
	--env tensorflow-1.8 \
	"python -m e2c.train --num-layers 3 --num-units 16 --batch-size 32 --max-steps 5000 --input-dir /data --output-dir /output --model-dir /output/model"