# python custom_dan_EU.py -l 2 -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -j 16 --log custom_dan/Restnet152/MKME/Batch_16/Label_2 --per-class-eval
# python custom_dan_EU.py -l 2 -a resnet152 -lf MKME --epochs 150 -b 16 -i 500 -j 16 --log custom_dan/Restnet152/MKME/Batch_16/Label_2 --per-class-eval --phase analysis
# python custom_dan_EU.py -l 2 -a resnet152 -lf MKMMD --epochs 150 -b 16 -i 500 -j 16 --log custom_dan/Restnet152/MKMMD/Batch_16/Label_2 --per-class-eval
# python custom_dan_EU.py -l 2 -a resnet152 -lf MKMMD --epochs 150 -b 16 -i 500 -j 16 --log custom_dan/Restnet152/MKMMD/Batch_16/Label_2 --per-class-eval --phase analysis

# python custom_dan_EU.py -d Capture -a resnet50 -lf MKMMD --epochs 150 -b 16 -i 500 -j 16 --log custom_dan/New_data/Resnet50/MKMMD/Batch_16/ --per-class-eval
# python custom_dan_EU.py -d Capture -a resnet50 -lf MKMMD --epochs 150 -b 16 -i 500 -j 16 --log custom_dan/New_data/Resnet50/MKMMD/Batch_16/ --per-class-eval --phase analysis
# python custom_dan_EU.py -d Capture -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -j 16 --log custom_dan/New_data/Resnet50/MKME/Batch_16/ --per-class-eval
# python custom_dan_EU.py -d Capture -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -j 16 --log custom_dan/New_data/Resnet50/MKME/Batch_16/ --per-class-eval --phase analysis

# python custom_dan_EU.py -d Capture -a resnet50 -lf MKMMD --epochs 150 -b 16 -i 500 -j 16 --trade-off 0 --log custom_dan/New_data/Resnet50/MKMMD/Batch_16/Trade_off_0 --per-class-eval
# python custom_dan_EU.py -d Capture -a resnet50 -lf MKMMD --epochs 150 -b 16 -i 500 -j 16 --trade-off 0 --log custom_dan/New_data/Resnet50/MKMMD/Batch_16/Trade_off_0 --per-class-eval --phase analysis

# python custom_dan_EU.py -d Both -a resnet50 -lf MKMMD --epochs 150 -b 16 -i 500 --log custom_dan/Both/Resnet50/MKMMD/Batch_16/ --per-class-eval
# python custom_dan_EU.py -d Both -a resnet50 -lf MKMMD --epochs 150 -b 16 -i 500 --log custom_dan/Both/Resnet50/MKMMD/Batch_16/ --per-class-eval --phase analysis

# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb S2T --trade-off 1 --log custom_dan/Both/Resnet50/MKME/Batch_16/S2T/one --per-class-eval
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb S2T --trade-off 1 --log custom_dan/Both/Resnet50/MKME/Batch_16/S2T/one --per-class-eval --phase analysis
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb S2T --trade-off 0 --log custom_dan/Both/Resnet50/MKME/Batch_16/S2T/zero --per-class-eval
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb S2T --trade-off 0 --log custom_dan/Both/Resnet50/MKME/Batch_16/S2T/zero --per-class-eval --phase analysis

# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb T2S --trade-off 1 --log custom_dan/Both/Resnet50/MKME/Batch_16/T2S/one --per-class-eval
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb T2S --trade-off 1 --log custom_dan/Both/Resnet50/MKME/Batch_16/T2S/one --per-class-eval --phase analysis
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb T2S --trade-off 0 --log custom_dan/Both/Resnet50/MKME/Batch_16/T2S/zero --per-class-eval
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb T2S --trade-off 0 --log custom_dan/Both/Resnet50/MKME/Batch_16/T2S/zero --per-class-eval --phase analysis

# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb S2T --trade-off 1 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/S2T/one --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb S2T --trade-off 1 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/S2T/one --per-class-eval --phase analysis
# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb S2T --trade-off 0 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/S2T/zero --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb S2T --trade-off 0 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/S2T/zero --per-class-eval --phase analysis

# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb T2S --trade-off 1 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/T2S/one --per-class-eval
# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb T2S --trade-off 1 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/T2S/one --per-class-eval --phase analysis

# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb S2T --trade-off 1 --log custom_dan/Both/Resnet50/MKME/Batch_16/S2T/one --per-class-eval --phase test
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb S2T --trade-off 0 --log custom_dan/Both/Resnet50/MKME/Batch_16/S2T/zero --per-class-eval --phase test
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb T2S --trade-off 1 --log custom_dan/Both/Resnet50/MKME/Batch_16/T2S/one --per-class-eval --phase test
# python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb T2S --trade-off 0 --log custom_dan/Both/Resnet50/MKME/Batch_16/T2S/zero --per-class-eval --phase test
# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb T2S --trade-off 1 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/T2S/one --per-class-eval --phase test
# python custom_dan.py -d Both -a resnet50 -lf MKME --epochs 150 -b 16 -i 500 -kb S2T --trade-off 1 --log custom_dan/Both/Mahanalobis/Resnet50/MKME/Batch_16/S2T/one --per-class-eval --phase test

python custom_dan_EU.py -d Both -a resnet50 -lf MKME --epochs 2 -b 16 -i 1 -kb S2T --trade-off 1 --log Test/ --per-class-eval