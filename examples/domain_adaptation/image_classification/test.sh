# python dan.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 3 --seed 1 -i 1 --log Test/
# python dan.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 3 --seed 1 -i 1 --log Test/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm -ts none --byte-size 32 --epochs 3 -b 16 -i 1 -scenario T2S --trade-off 0 --log Test/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --byte-size 32 --epochs 3 -b 8 -i 10 -scenario S2T --trade-off 1 --log Test/ --per-class-eval
# python custom_dan.py -d Both -a lenet -lf MKMMD -ts none --byte-size 32 --epochs 3 -b 8 -i 10 -scenario S2T --trade-off 1 --log Test/ --per-class-eval
# python custom_dan.py -d Both -a vgg11 -lf MKMMD -ts none --byte-size 32 --epochs 3 -b 8 -i 10 -scenario S2T --trade-off 1 --log Test/ --per-class-eval

# python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --byte-size 32 --epochs 3 -b 8 -i 10 -scenario S2T --trade-off 1 --log Test/ --per-class-eval
# python custom_dan.py -d Both -a lenet -lf MKMMD -ts none --byte-size 32 --epochs 3 -b 8 -i 10 -scenario S2T --trade-off 1 --log Test/ --per-class-eval
# python custom_dan.py -d Both -a vgg11 -lf MKMMD -ts none --byte-size 32 --epochs 3 -b 8 -i 10 -scenario S2T --trade-off 1 --log Test/ --per-class-eval


# python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm -ts none --byte-size 256 --epochs 3 -b 16 -i 1 -scenario T2S --trade-off 0 --log Test/ --per-class-eval


###################################################################
# #!/bin/bash

# # Read the content of model_list.txt and store it in a variable
# model_list=$(cat model_list.txt)

# # Set the internal field separator (IFS) to comma to split the models
# IFS=','

# # Convert the model list into an array
# read -ra models <<< "$model_list"

# # Loop through the models and process each one
# for model in "${models[@]}"; do
#     echo "Processing model: $model"
#     python custom_dan.py -d Both -a $model -lf MKMMD -ts none --byte-size 256 --epochs 2 -b 8 -i 2 -scenario S2T --trade-off 1 --log Test/ --per-class-eval
#     # Add your code here to process each model
# done

# # Reset IFS to its default value
# IFS=$' \t\n'


# python custom_dan.py -d Both -a resnet50 -lf None -ts none --epochs 2 -b 16 -i 2 -scenario S2T --trade-off 0 --log Test/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKME -ts pinverse --epochs 2 -b 16 -i 2 -scenario S2T -ss 1 --trade-off 0.5 --log Test/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKME -ts pinverse --epochs 2 -b 16 -i 2 -scenario S2T -ss 2 --trade-off 0.5 --log Test/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKME -ts pinverse --epochs 2 -b 16 -i 2 -scenario S2T -ss 3 --trade-off 0.5 --log Test/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 3 -b 4 -i 1 --byte-size 256 -ss none -scenario S2T -per 0 --trade-off 1 --log Test/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 3 -b 4 -i 1 --byte-size 256 -ss none -scenario S2T -per 0 --trade-off 1 --log Test/ --per-class-eval --phase analysis

# python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 2 -b 8 -i 2 --byte-size 256 -ss none -scenario S2T --trade-off 0 -per 0 --log Result/DAN/Lambda_zero/ --per-class-eval