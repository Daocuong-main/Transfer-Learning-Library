python dan.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 3 --seed 1 -i 10 --log Test/
python dan.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 3 --seed 1 -i 10 --log Test/ --per-class-eval
python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 3 -b 16 -i 10 -scenario T2S --trade-off 0.2 --log Test/ --per-class-eval