#!/usr/bin/env bash
# ResNet50
CUDA_VISIBLE_DEVICES=0 python custom_dan.py -l 0 -a resnet50 -j 70 --epochs 100 -b 64 -i 1 --log custom_dan/label_0
CUDA_VISIBLE_DEVICES=0 python custom_dan.py -l 1 -a resnet50 -j 70 --epochs 100 -b 64 -i 1 --log custom_dan/label_1
CUDA_VISIBLE_DEVICES=0 python custom_dan.py -l 2 -a resnet50 -j 70 --epochs 100 -b 64 -i 1 --log custom_dan/label_2
CUDA_VISIBLE_DEVICES=0 python custom_dan.py -l 3 -a resnet50 -j 70 --epochs 100 -b 64 -i 1 --log custom_dan/label_3

CUDA_VISIBLE_DEVICES=0 python custom_dan.py -l 0 -a resnet50 -j 70 --epochs 100 -b 64 -i 100 --log custom_dan/label_0
CUDA_VISIBLE_DEVICES=0 python custom_dan.py -l 1 -a resnet50 -j 70 --epochs 100 -b 64 -i 100 --log custom_dan/label_1
CUDA_VISIBLE_DEVICES=0 python custom_dan.py -l 2 -a resnet50 -j 70 --epochs 100 -b 64 -i 100 --log custom_dan/label_2
CUDA_VISIBLE_DEVICES=0 python custom_dan.py -l 3 -a resnet50 -j 70 --epochs 100 -b 64 -i 100 --log custom_dan/label_3