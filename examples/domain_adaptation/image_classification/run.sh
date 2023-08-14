#!/bin/bash
# python custom_dan_EU.py -l 2 -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -j 16 --log custom_dan/Restnet152/MKME/Batch_16/Label_2 --per-class-eval
# python custom_dan_EU.py -l 2 -a resnet152 -lf MKME --epochs 500 -b 16 -i 500 -j 16 --log custom_dan/Restnet152/MKME/Batch_16/Label_2 --per-class-eval --phase analysis
# python custom_dan_EU.py -l 2 -a resnet152 -lf MKMMD --epochs 500 -b 16 -i 500 -j 16 --log custom_dan/Restnet152/MKMMD/Batch_16/Label_2 --per-class-eval
# python custom_dan_EU.py -l 2 -a resnet152 -lf MKMMD --epochs 500 -b 16 -i 500 -j 16 --log custom_dan/Restnet152/MKMMD/Batch_16/Label_2 --per-class-eval --phase analysis

# python custom_dan_EU.py -d Capture -a resnet50 -lf MKMMD --epochs 500 -b 16 -i 500 -j 16 --log custom_dan/New_data/Resnet50/MKMMD/Batch_16/ --per-class-eval
# python custom_dan_EU.py -d Capture -a resnet50 -lf MKMMD --epochs 500 -b 16 -i 500 -j 16 --log custom_dan/New_data/Resnet50/MKMMD/Batch_16/ --per-class-eval --phase analysis
# python custom_dan_EU.py -d Capture -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -j 16 --log custom_dan/New_data/Resnet50/MKME/Batch_16/ --per-class-eval
# python custom_dan_EU.py -d Capture -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -j 16 --log custom_dan/New_data/Resnet50/MKME/Batch_16/ --per-class-eval --phase analysis

# python custom_dan_EU.py -d Capture -a resnet50 -lf MKMMD --epochs 500 -b 16 -i 500 -j 16 --trade-off 0 --log custom_dan/New_data/Resnet50/MKMMD/Batch_16/Trade_off_0 --per-class-eval
# python custom_dan_EU.py -d Capture -a resnet50 -lf MKMMD --epochs 500 -b 16 -i 500 -j 16 --trade-off 0 --log custom_dan/New_data/Resnet50/MKMMD/Batch_16/Trade_off_0 --per-class-eval --phase analysis

# python custom_dan_EU.py -d Both -a resnet50 -lf MKMMD --epochs 500 -b 16 -i 500 --log custom_dan/Both/Resnet50/MKMMD/Batch_16/ --per-class-eval
# python custom_dan_EU.py -d Both -a resnet50 -lf MKMMD --epochs 500 -b 16 -i 500 --log custom_dan/Both/Resnet50/MKMMD/Batch_16/ --per-class-eval --phase analysis

# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb S2T --trade-off 1 --log custom_dan/Both/Resnet50/MKME/Batch_16/S2T/one --per-class-eval
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb S2T --trade-off 1 --log custom_dan/Both/Resnet50/MKME/Batch_16/S2T/one --per-class-eval --phase analysis
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb S2T --trade-off 0 --log custom_dan/Both/Resnet50/MKME/Batch_16/S2T/zero --per-class-eval
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb S2T --trade-off 0 --log custom_dan/Both/Resnet50/MKME/Batch_16/S2T/zero --per-class-eval --phase analysis

# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb T2S --trade-off 1 --log custom_dan/Both/Resnet50/MKME/Batch_16/T2S/one --per-class-eval
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb T2S --trade-off 1 --log custom_dan/Both/Resnet50/MKME/Batch_16/T2S/one --per-class-eval --phase analysis
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb T2S --trade-off 0 --log custom_dan/Both/Resnet50/MKME/Batch_16/T2S/zero --per-class-eval
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb T2S --trade-off 0 --log custom_dan/Both/Resnet50/MKME/Batch_16/T2S/zero --per-class-eval --phase analysis

# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb S2T --trade-off 1 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/S2T/one --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb S2T --trade-off 1 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/S2T/one --per-class-eval --phase analysis
# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb S2T --trade-off 0 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/S2T/zero --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb S2T --trade-off 0 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/S2T/zero --per-class-eval --phase analysis

# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb T2S --trade-off 1 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/T2S/one --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb T2S --trade-off 1 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/T2S/one --per-class-eval --phase analysis

# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb S2T --trade-off 1 --log custom_dan/Both/Resnet50/MKME/Batch_16/S2T/one --per-class-eval --phase test
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb S2T --trade-off 0 --log custom_dan/Both/Resnet50/MKME/Batch_16/S2T/zero --per-class-eval --phase test
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb T2S --trade-off 1 --log custom_dan/Both/Resnet50/MKME/Batch_16/T2S/one --per-class-eval --phase test
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb T2S --trade-off 0 --log custom_dan/Both/Resnet50/MKME/Batch_16/T2S/zero --per-class-eval --phase test
# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb T2S --trade-off 1 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/T2S/one --per-class-eval --phase test
# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb S2T --trade-off 1 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/S2T/one --per-class-eval --phase test

# python custom_dan_EU.py -d Both -a resnet50 -lf MKMMD --epochs 500 -b 16 -i 500 -kb S2T --trade-off 1 --log Test/MKMMD --per-class-eval
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb S2T --trade-off 1 --log Test/non_cov --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 500 -b 16 -i 500 -kb S2T --trade-off 1 --log Test/pinverse --per-class-eval

#1
# python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 500 -b 16 -i 500 -scenario S2T --trade-off 1 --log Test/S2T/SCF/unnorm/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 500 -b 16 -i 500 -scenario S2T --trade-off 1 --log Test/S2T/SCF/pinverse/ --per-class-eval

# python custom_dan.py -d Both -a resnet50 -lf MKME -ts unnorm --epochs 500 -b 16 -i 500 -scenario S2T --trade-off 1 --log Test/S2T/ME/unnorm/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKME -ts pinverse --epochs 500 -b 16 -i 500 -scenario S2T --trade-off 1 --log Test/S2T/ME/pinverse/ --per-class-eval

# python custom_dan.py -d Both -a resnet50 -lf MKME -ts unnorm --epochs 500 -b 16 -i 500 -scenario S2T --trade-off 0 --log Test/S2T/zero_lambda/ --per-class-eval

# python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 500 -b 16 -i 500 -scenario T2S --trade-off 1 --log Test/T2S/SCF/unnorm/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 500 -b 16 -i 500 -scenario T2S --trade-off 1 --log Test/T2S/SCF/pinverse/ --per-class-eval

# python custom_dan.py -d Both -a resnet50 -lf MKME -ts unnorm --epochs 500 -b 16 -i 500 -scenario T2S --trade-off 1 --log Test/T2S/ME/unnorm/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKME -ts pinverse --epochs 500 -b 16 -i 500 -scenario T2S --trade-off 1 --log Test/T2S/ME/pinverse/ --per-class-eval

# python custom_dan.py -d Both -a resnet50 -lf MKME -ts unnorm --epochs 500 -b 16 -i 500 -scenario T2S --trade-off 0 --log Test/T2S/zero_lambda/ --per-class-eval


# # for ((iter=1; iter<=1; iter++))
# # do
# #    for i in 0.1 2
# #    do
# #        python custom_dan.py -d Both -a resnet50 -lf MKMMD --epochs 250 -b 16 -i 500 -j 16 -scenario S2T --trade-off $i --log Test/Change_lambda/S2T/MKMMD/$i/ --per-class-eval
# #    done
# # done

# # for ((iter=1; iter<=5; iter++))
# # do
# #    for i in 1
# #    do
# #        python custom_dan.py -d Both -a resnet50 -lf MKMMD --epochs 250 -b 16 -i 500 -j 16 -scenario S2T --trade-off $i --log Test/Change_lambda/S2T/MKMMD/$i/ --per-class-eval
# #    done
# # done

# # for ((iter=1; iter<=1; iter++))
# # do
# #     for i in 0.1 0.2
# #     do
# #         python custom_dan.py -d Both -a resnet50 -lf MKMMD --epochs 250 -b 16 -i 500 -j 16 -scenario T2S --trade-off $i --log Test/Change_lambda/T2S/MKMMD/$i/ --per-class-eval
# #     done
# # done

# # for ((iter=1; iter<=2; iter++))
# # do
# #     for i in 2
# #     do
# #         python custom_dan.py -d Both -a resnet50 -lf MKMMD --epochs 250 -b 16 -i 500 -j 16 -scenario T2S --trade-off $i --log Test/Change_lambda/T2S/MKMMD/$i/ --per-class-eval
# #     done
# # done

# for ((iter=1; iter<=5; iter++))
# do
#     for i in 5
#     do
#         python custom_dan.py -d Both -a resnet50 -lf MKMMD --epochs 250 -b 16 -i 500 -j 16 -scenario T2S --trade-off $i --log Test/Change_lambda/T2S/MKMMD/$i/ --per-class-eval
#     done
# done
# for ((iter=1; iter<=5; iter++))
# do
#     for i in 1
#     do
#         python custom_dan.py -d Both -a resnet50 -lf MKMMD --epochs 250 -b 16 -i 500 -j 16 -scenario T2S --trade-off $i --log Test/Change_lambda/T2S/MKMMD/$i/ --per-class-eval
#     done
# done

# Thay doi lambda
# for ((iter=1; iter<=3; iter++))
# do
#     for i in 0.05 0.2 0.5 2 5
#     do
#         python custom_dan.py -d Both -a resnet50 -lf MKME -ts pinverse --epochs 250 -b 16 -i 500 -scenario S2T --trade-off $i --log Test/Change_lambda/S2T/pinverse/ME/$i/ --per-class-eval
#     done
#     for i in 0.05 0.1 0.2 0.5 2 5
#     do
#         python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 250 -b 16 -i 500 -scenario S2T --trade-off $i --log Test/Change_lambda/S2T/pinverse/SCF/$i/ --per-class-eval
#     done
#     for i in 0.05 0.1 0.5 2 5
#     do
#         python custom_dan.py -d Both -a resnet50 -lf MKME -ts unnorm --epochs 250 -b 16 -i 500 -scenario S2T --trade-off $i --log Test/Change_lambda/S2T/unnorm/ME/$i/ --per-class-eval
#     done
#     for i in 0.05 0.1 0.2 0.5 5
#     do
#         python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 250 -b 16 -i 500 -scenario S2T --trade-off $i --log Test/Change_lambda/S2T/unnorm/SCF/$i/ --per-class-eval
#     done
# done

# for i in 0.05 0.1 0.5 2 5
# do
#     python custom_dan.py -d Both -a resnet50 -lf MKME -ts pinverse --epochs 250 -b 16 -i 500 -scenario T2S --trade-off $i --log Test/Change_lambda/T2S/pinverse/ME/$i/ --per-class-eval
# done
# for i in 0.05 0.1 0.2 0.5 2 5
# do
#     python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 250 -b 16 -i 500 -scenario T2S --trade-off $i --log Test/Change_lambda/T2S/pinverse/SCF/$i/ --per-class-eval
# done
# for i in 0.1 0.2 0.5 2 5
# do
#     python custom_dan.py -d Both -a resnet50 -lf MKME -ts unnorm --epochs 250 -b 16 -i 500 -scenario T2S --trade-off $i --log Test/Change_lambda/T2S/unnorm/ME/$i/ --per-class-eval
# done
# for i in 0.05 0.1 0.5 2 5
# do
#     python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 250 -b 16 -i 500 -scenario T2S --trade-off $i --log Test/Change_lambda/T2S/unnorm/SCF/$i/ --per-class-eval
# done
    
    

    
    

# for ((iter=1; iter<=1; iter++))
# do
#     python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 250 -b 16 -i 500 -scenario S2T --trade-off 2 --log Test/Change_lambda/S2T/unnorm/SCF/2/ --per-class-eval
#     python custom_dan.py -d Both -a resnet50 -lf MKME -ts unnorm --epochs 250 -b 16 -i 500 -scenario S2T --trade-off 0.2 --log Test/Change_lambda/S2T/unnorm/ME/0.2/ --per-class-eval
#     python custom_dan.py -d Both -a resnet50 -lf MKME -ts pinverse --epochs 250 -b 16 -i 500 -scenario S2T --trade-off 0.1 --log Test/Change_lambda/S2T/pinverse/ME/0.1/ --per-class-eval
# done

# python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 250 -b 16 -i 500 -scenario T2S --trade-off 0.2 --log Test/Change_lambda/T2S/unnorm/SCF/0.2/ --per-class-eval

# python custom_dan.py -d Both -a resnet50 -lf MKME -ts unnorm --epochs 250 -b 16 -i 500 -scenario T2S --trade-off 0.05 --log Test/Change_lambda/T2S/unnorm/ME/0.05/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKME -ts pinverse --epochs 250 -b 16 -i 500 -scenario T2S --trade-off 0.2 --log Test/Change_lambda/T2S/pinverse/ME/0.2/ --per-class-eval


# doi byte
for i in 512
do
    python custom_dan.py -d nondan -a resnet50 -lf MKMMD -ts none --byte-size $i --epochs 150 -b 8 -i 300 -j 16 -scenario S2T --trade-off 0 --log Test/Change_byte_new/S2T/ --per-class-eval
    python custom_dan.py -d nondan -a resnet50 -lf MKMMD -ts none --byte-size $i --epochs 150 -b 8 -i 300 -j 16 -scenario T2S --trade-off 0 --log Test/Change_byte_new/T2S/ --per-class-eval
done

# python custom_dan.py -d Both -a resnet50 -lf MKME -ts pinverse --epochs 250 -b 16 -i 500 -scenario S2T --trade-off 0.5 --log Test/Change_lambda/S2T/pinverse/ME/0.5/ --per-class-eval --phase analysis
# python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 250 -b 16 -i 500 -scenario S2T --trade-off 0.1 --log Test/Change_lambda/S2T/pinverse/SCF/0.1/ --per-class-eval --phase analysis
# python custom_dan.py -d Both -a resnet50 -lf MKME -ts unnorm --epochs 250 -b 16 -i 500 -scenario S2T --trade-off 0.2 --log Test/Change_lambda/S2T/unnorm/ME/0.2/ --per-class-eval --phase analysis
# python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 250 -b 16 -i 500 -scenario S2T --trade-off 2 --log Test/Change_lambda/S2T/unnorm/SCF/2/ --per-class-eval --phase analysis
# python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 16 -i 500 -scenario S2T --trade-off 5 --log Test/Change_lambda/S2T/MKMMD/5/ --per-class-eval --phase analysis

# python custom_dan.py -d Both -a resnet50 -lf MKME -ts pinverse --epochs 250 -b 16 -i 500 -scenario T2S --trade-off 0.2 --log Test/Change_lambda/T2S/pinverse/ME/0.2/ --per-class-eval --phase analysis
# python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 250 -b 16 -i 500 -scenario T2S --trade-off 0.1 --log Test/Change_lambda/T2S/pinverse/SCF/0.1/ --per-class-eval --phase analysis
# python custom_dan.py -d Both -a resnet50 -lf MKME -ts unnorm --epochs 250 -b 16 -i 500 -scenario T2S --trade-off 0.1 --log Test/Change_lambda/T2S/unnorm/ME/0.1/ --per-class-eval --phase analysis
# python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 250 -b 16 -i 500 -scenario T2S --trade-off 0.2 --log Test/Change_lambda/T2S/unnorm/SCF/0.2/ --per-class-eval --phase analysis
# python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 16 -i 500 -scenario T2S --trade-off 0.5 --log Test/Change_lambda/T2S/MKMMD/0.5/ --per-class-eval --phase analysis

# python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 16 -i 500 -scenario S2T --trade-off 5 --log Test/Change_lambda/S2T/pinverse/ME/5/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 16 -i 500 -scenario S2T --trade-off 5 --log Test/Change_lambda/S2T/pinverse/ME/5/ --per-class-eval --phase analysis