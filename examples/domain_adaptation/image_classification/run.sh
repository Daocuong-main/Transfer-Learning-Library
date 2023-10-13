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
# for i in 512
# do
#     python custom_dan.py -d nondan -a resnet50 -lf MKMMD -ts none --byte-size $i --epochs 150 -b 8 -i 300 -j 16 -scenario S2T --trade-off 0 --log Test/Change_byte_new/S2T/ --per-class-eval
#     python custom_dan.py -d nondan -a resnet50 -lf MKMMD -ts none --byte-size $i --epochs 150 -b 8 -i 300 -j 16 -scenario T2S --trade-off 0 --log Test/Change_byte_new/T2S/ --per-class-eval
# done

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

# python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 1 -b 16 -i 93 -scenario S2T --trade-off 2 --log Test/Change_lambda/S2T/unnorm/SCF/2/ --per-class-eval --phase test
# python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 16 -i 500 -scenario S2T --trade-off 5 --log Test/Change_lambda/S2T/MKMMD/5/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 16 -i 500 -scenario S2T --trade-off 5 --log Test/Change_lambda/S2T/MKMMD/5/ --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 1 -b 16 -i 93 -scenario T2S --trade-off 0.2 --log Test/Change_lambda/T2S/unnorm/SCF/0.2/ --per-class-eval --phase test
# python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 1 -b 16 -i 93 -scenario T2S --trade-off 0.5 --log Test/Change_lambda/T2S/MKMMD/0.5/ --per-class-eval --phase test


# python custom_dan.py -d nondan -a resnet50 -lf None -ts none --epochs 250 -b 4 -i 500 -scenario S2T --byte-size 256 --trade-off 0 --log Test/Change_backbone/S2T/ --per-class-eval
# python custom_dan.py -d nondan -a vgg11 -lf None -ts none --epochs 250 -b 4 -i 500 -scenario S2T --byte-size 256 --trade-off 0 --log Test/Change_backbone/S2T/ --per-class-eval
# python custom_dan.py -d nondan -a resnet18 -lf None -ts none --epochs 250 -b 4 -i 500 -scenario S2T --byte-size 256 --trade-off 0 --log Test/Change_backbone/S2T/ --per-class-eval
# python custom_dan.py -d nondan -a darknet53 -lf None -ts none --epochs 250 -b 4 -i 500 -scenario S2T --byte-size 256 --trade-off 0 --log Test/Change_backbone/S2T/ --per-class-eval
# python custom_dan.py -d nondan -a mobilevitv2_050 -lf None -ts none --epochs 250 -b 4 -i 500 -scenario S2T --byte-size 256 --trade-off 0 --log Test/Change_backbone/S2T/ --per-class-eval

# python custom_dan.py -d nondan -a resnet50 -lf None -ts none --epochs 250 -b 4 -i 500 -scenario T2S --byte-size 256 --trade-off 0 --log Test/Change_backbone/T2S/ --per-class-eval
# python custom_dan.py -d nondan -a vgg11 -lf None -ts none --epochs 250 -b 4 -i 500 -scenario T2S --byte-size 256 --trade-off 0 --log Test/Change_backbone/T2S/ --per-class-eval
# python custom_dan.py -d nondan -a resnet18 -lf None -ts none --epochs 250 -b 4 -i 500 -scenario T2S --byte-size 256 --trade-off 0 --log Test/Change_backbone/T2S/ --per-class-eval
# python custom_dan.py -d nondan -a darknet53 -lf None -ts none --epochs 250 -b 4 -i 500 -scenario T2S --byte-size 256 --trade-off 0 --log Test/Change_backbone/T2S/ --per-class-eval
# python custom_dan.py -d nondan -a mobilevitv2_050 -lf None -ts none --epochs 250 -b 4 -i 500 -scenario T2S --byte-size 256 --trade-off 0 --log Test/Change_backbone/T2S/ --per-class-eval

# python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 250 -b 16 -i 500 -scenario S2T --trade-off 2 --log Test/Change_lambda/S2T/unnorm/SCF/2/ --per-class-eval --phase analysis
# python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 16 -i 500 -scenario S2T --trade-off 5 --log Test/Change_lambda/S2T/MKMMD/5/ --per-class-eval --phase analysis
# python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 250 -b 16 -i 500 -scenario T2S --trade-off 0.2 --log Test/Change_lambda/T2S/unnorm/SCF/0.2/ --per-class-eval --phase analysis
# python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 16 -i 500 -scenario T2S --trade-off 0.5 --log Test/Change_lambda/T2S/MKMMD/0.5/ --per-class-eval --phase analysis


# Normal classification
# for i in 32 256 512
# do
#     python custom_dan.py -d nondan -a resnet50 -lf None -ts none --byte-size $i --epochs 250 -b 4 -i 1 -j 8 -ss none -scenario S2T --trade-off 0 --log Result/No_TF/$i/ --per-class-eval
#     python custom_dan.py -d nondan -a mobilevitv2_050 -lf None -ts none --byte-size $i --epochs 250 -b 4 -i 1 -j 8 -ss none -scenario S2T --trade-off 0 --log Result/No_TF/$i/ --per-class-eval
#     python custom_dan.py -d nondan -a darknet53 -lf None -ts none --byte-size $i --epochs 250 -b 4 -i 1 -j 8 -ss none -scenario S2T --trade-off 0 --log Result/No_TF/$i/ --per-class-eval
# done

# # DAN
# for ((iter=1; iter<=1; iter++))
# do
#     for i in 512
#     do
#         for b in 2 3
#         do
#             for lambda in 0.05 0.5 1 2
#             do
#                 python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 250 -b 4 -i 500 --byte-size $i -ss $b -scenario S2T -per 0 --trade-off $lambda --log Result/DAN/Byte_$i/SCF-unnorm/Test_$b/Lambda_$lambda/ --per-class-eval
#                 python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 250 -b 4 -i 500 --byte-size $i -ss $b -scenario S2T -per 0 --trade-off $lambda --log Result/DAN/Byte_$i/SCF-pinverse/Test_$b/Lambda_$lambda/ --per-class-eval
#                 python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 4 -i 500 --byte-size $i -ss $b -scenario S2T -per 0 --trade-off $lambda --log Result/DAN/Byte_$i/MKMMD/Test_$b/Lambda_$lambda/ --per-class-eval
                
#                 python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 250 -b 4 -i 500 --byte-size $i -ss $b -scenario S2T -per 0 --trade-off $lambda --log Result/DAN/Byte_$i/SCF-unnorm/Test_$b/Lambda_$lambda/ --per-class-eval --phase analysis
#                 python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 250 -b 4 -i 500 --byte-size $i -ss $b -scenario S2T -per 0 --trade-off $lambda --log Result/DAN/Byte_$i/SCF-pinverse/Test_$b/Lambda_$lambda/ --per-class-eval --phase analysis
#                 python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 4 -i 500 --byte-size $i -ss $b -scenario S2T -per 0 --trade-off $lambda --log Result/DAN/Byte_$i/MKMMD/Test_$b/Lambda_$lambda/ --per-class-eval --phase analysis
#             done
#         done
#     done
# done

# Define a file to store variable values
# output_file="variable_values.txt"

# for ((iter=1; iter<=5; iter++))
# do
#     for i in 512
#     do
#         for lambda in 0.05 0.5 1 2
#         do
#             # Write the current variable values to the output file
#             echo "iter: $iter, i: $i, lambda: $lambda" >> "$output_file"
            
#             # Rest of your code here
#             python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 250 -b 4 -i 500 --byte-size $i -ss none -scenario S2T --trade-off $lambda --log Result/DAN/Byte_$i/SCF-unnorm/Test_none/Lambda_$lambda/ --per-class-eval
#             python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 250 -b 4 -i 500 --byte-size $i -ss none -scenario S2T --trade-off $lambda --log Result/DAN/Byte_$i/SCF-pinverse/Test_none/Lambda_$lambda/ --per-class-eval
#             # python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 8 -i 500 --byte-size $i -ss none -scenario S2T --trade-off $lambda --log Result/DAN/Byte_$i/MKMMD/Test_none/Lambda_$lambda/ --per-class-eval
            
#             python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 250 -b 4 -i 500 --byte-size $i -ss none -scenario S2T --trade-off $lambda --log Result/DAN/Byte_$i/SCF-unnorm/Test_none/Lambda_$lambda/ --per-class-eval --phase analysis
#             python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 250 -b 4 -i 500 --byte-size $i -ss none -scenario S2T --trade-off $lambda --log Result/DAN/Byte_$i/SCF-pinverse/Test_none/Lambda_$lambda/ --per-class-eval --phase analysis
#             # python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 8 -i 500 --byte-size $i -ss none -scenario S2T --trade-off $lambda --log Result/DAN/Byte_$i/MKMMD/Test_none/Lambda_$lambda/ --per-class-eval --phase analysis
#         done
#     done
# done

# output_file="variable_values.txt"
# for ((iter=1; iter<=1; iter++))
# do
# for b in 256
#     do
#         for i in 0.05 1
#         do
#             for k in 0 0.05 0.1 0.5 1
#             do
#                 # Write the current variable values to the output file
#                 echo "iter: $iter, byte $b ,lambda: $i, use_$k percent" >> "$output_file"
#                 python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 50 -b 8 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/MKMMD/lambda_$i/use_$k/ --per-class-eval
#                 python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 50 -b 4 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/SCF_unnorm/lambda_$i/use_$k/ --per-class-eval
#                 python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 50 -b 4 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/SCF_pinverse/lambda_$i/use_$k/ --per-class-eval
#                 python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 50 -b 8 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/MKMMD/lambda_$i/use_$k/ --per-class-eval --phase analysis
#                 python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 50 -b 4 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/SCF_unnorm/lambda_$i/use_$k/ --per-class-eval --phase analysis
#                 python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 50 -b 4 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/SCF_pinverse/lambda_$i/use_$k/ --per-class-eval --phase analysis
#             done
#         done
#     done
# done

output_file="variable_values.txt"
for ((iter=1; iter<=1; iter++))
do
for b in 512
    do
        for i in 0.05
        do
            # for k in 0.05 0.1 0.5 1
            for k in 0.5
            do
                # Write the current variable values to the output file
                echo "iter: $iter, byte $b ,lambda: $i, use_$k percent" >> "$output_file"
                python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 50 -b 8 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/MKMMD/lambda_$i/use_$k/ --per-class-eval
                # python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 50 -b 4 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/SCF_unnorm/lambda_$i/use_$k/ --per-class-eval
                # python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 50 -b 4 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/SCF_pinverse/lambda_$i/use_$k/ --per-class-eval
                # python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 50 -b 8 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/MKMMD/lambda_$i/use_$k/ --per-class-eval --phase analysis
                # python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 50 -b 4 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/SCF_unnorm/lambda_$i/use_$k/ --per-class-eval --phase analysis
                # python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 50 -b 4 -i 500 --byte-size $b -ss none -scenario S2T --trade-off $i -per $k --log Result/DAN/percent/byte_$b/SCF_pinverse/lambda_$i/use_$k/ --per-class-eval --phase analysis
            done
        done
    done
done

# python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 8 -i 500 --byte-size 256 -ss none -scenario S2T --trade-off 0 -per 0 --log Result/DAN/Lambda_zero/ --per-class-eval

# #Define a file to store variable values
# output_file="variable_values.txt"

# for ((iter=1; iter<=5; iter++))
# do
#     for i in 512
#     do
#         for lambda in 0.05
#         do
#             # Write the current variable values to the output file
#             echo "iter: $iter, i: $i, lambda: $lambda" >> "$output_file"
            
#             # Rest of your code here
#             python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 250 -b 4 -i 500 --byte-size $i -ss none -scenario S2T --trade-off $lambda --log Result/DAN/Byte_$i/SCF-unnorm/Test_none/Lambda_$lambda/ --per-class-eval
#             # python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 250 -b 4 -i 500 --byte-size $i -ss none -scenario S2T --trade-off $lambda --log Result/DAN/Byte_$i/SCF-pinverse/Test_none/Lambda_$lambda/ --per-class-eval
#             # python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 8 -i 500 --byte-size $i -ss none -scenario S2T --trade-off $lambda --log Result/DAN/Byte_$i/MKMMD/Test_none/Lambda_$lambda/ --per-class-eval
            
#             # python custom_dan.py -d Both -a resnet50 -lf SCF -ts unnorm --epochs 250 -b 4 -i 500 --byte-size $i -ss none -scenario S2T --trade-off $lambda --log Result/DAN/Byte_$i/SCF-unnorm/Test_none/Lambda_$lambda/ --per-class-eval --phase analysis
#             # python custom_dan.py -d Both -a resnet50 -lf SCF -ts pinverse --epochs 250 -b 4 -i 500 --byte-size $i -ss none -scenario S2T --trade-off $lambda --log Result/DAN/Byte_$i/SCF-pinverse/Test_none/Lambda_$lambda/ --per-class-eval --phase analysis
#             # python custom_dan.py -d Both -a resnet50 -lf MKMMD -ts none --epochs 250 -b 8 -i 500 --byte-size $i -ss none -scenario S2T --trade-off $lambda --log Result/DAN/Byte_$i/MKMMD/Test_none/Lambda_$lambda/ --per-class-eval --phase analysis
#         done
#     done
# done