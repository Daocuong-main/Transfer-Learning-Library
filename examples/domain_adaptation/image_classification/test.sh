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

# python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 1 -b 32 -i 300 --byte-size 256 -ss none -scenario S2T --trade-off 1 -per 0 --log Test/ --per-class-eval
# python fixmatch_test.py data/office31 -d Office31 -s W -t A -a resnet50 --lr 0.001 --bottleneck-dim 256 -ub 96 --epochs 1 --seed 0 --log logs/fixmatch/Office31_W2A --per-class-eval
# python fixmatch_test.py data/concat_dataset -d Concatdata -s D1 -t D2 -a resnet50 --lr 0.001 --bottleneck-dim 256 -ub 96 --epochs 20 --i 500 --seed 0 --log logs/New_revision/ --per-class-eval
# python mcc_test.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 1 -i 2 --seed 2 --bottleneck-dim 1024 --log logs/mcc/Office31_W2A
# python fixmatch_test.py data/office31 -d Office31 -s W -t A -a resnet50 --lr 0.001 --bottleneck-dim 256 -ub 96 --epochs 1 --i 2 --seed 0 --log logs/fixmatch/Office31_W2A
# python mcc_test.py data/concat_dataset -d Concatdata -s D1 -t D2 -a resnet50 --epochs 20 -i 500 --seed 0 --bottleneck-dim 1024 --log logs/New_revision/ --per-class-eval
# python custom_fixmatch.py -d Both -a resnet50 --lr 0.001 --bottleneck-dim 256 -ub 96 --epochs 20 --seed 0 --log Test/ --per-class-eval -scenario S2T --byte-size 256 -ss none -b 32 -i 300 -ts none
# output_file="variable_values.txt"
# for ((iter=1; iter<=1; iter++))
# do
# for b in 512
#     do
#         for i in 0.05
#         do
#             # for k in 0.05 0.1 0.5 1
#             for k in 0.5
#             do
#                 # Write the current variable values to the output file
#                 echo "iter: $iter, byte $b ,lambda: $i, use_$k percent" >> "$output_file"
#                 python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 1 -b 8 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Test/ --per-class-eval
#                 # python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 50 -b 4 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/SCF_unnorm/lambda_$i/use_$k/ --per-class-eval
#                 # python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 50 -b 4 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/SCF_pinverse/lambda_$i/use_$k/ --per-class-eval
#                 # python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 50 -b 8 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/MKMMD/lambda_$i/use_$k/ --per-class-eval --phase analysis
#                 # python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 50 -b 4 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/SCF_unnorm/lambda_$i/use_$k/ --per-class-eval --phase analysis
#                 # python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 50 -b 4 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/SCF_pinverse/lambda_$i/use_$k/ --per-class-eval --phase analysis
#             done
#         done
#     done
# done

# python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 1 -b 4 -i 2 --byte-size 256 -ss none -scenario S2T --trade-off 1 --log Test/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 1 -b 2 -i 2 --byte-size 256 -ss none -scenario S2T --trade-off 1 --log Test/ --per-class-eval
python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 50 -b 32 -i 100 --byte-size 256 -ss none -scenario S2T --trade-off 1 --log Test/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 1 -b 32 -i 2 --byte-size 256 -ss none -scenario S2T --trade-off 1 --log Test/ --per-class-eval