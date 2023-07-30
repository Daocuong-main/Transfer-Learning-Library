# python dan.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 3 --seed 1 -i 1 --log Test/
# python dan.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 3 --seed 1 -i 1 --log Test/ --per-class-eval
python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --byte-size 256 --epochs 3 -b 16 -i 1 -scenario T2S --trade-off 0 --log Test/ --per-class-eval