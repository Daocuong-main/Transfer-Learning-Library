#!/usr/bin/env bash
# ResNet50
python custom_dan_EU.py -l 0 -a resnet50 -lf Both -s_param 1 -rf 10 --epochs 50 -b 64 -i 100 -j 70 --log custom_dan_ME_EU/Label_0
python custom_dan_EU.py -l 1 -a resnet50 -lf Both -s_param 1 -rf 10 --epochs 50 -b 64 -i 100 -j 70 --log custom_dan_ME_EU/Label_1
python custom_dan_EU.py -l 2 -a resnet50 -lf Both -s_param 1 -rf 10 --epochs 50 -b 64 -i 100 -j 70 --log custom_dan_ME_EU/Label_2
python custom_dan_EU.py -l 3 -a resnet50 -lf Both -s_param 1 -rf 10 --epochs 50 -b 64 -i 100 -j 70 --log custom_dan_ME_EU/Label_3