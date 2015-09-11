#!/bin/bash

# 15594564
#   200000
#
# 15794564

ntest=200000

SIZES="
15594564
10396376
6930917
4620611
3080407
2053605
1369070
912713
608475
405650
270433
180289
120192
80128
53418
35612
23741
15827
10551
7034
4689
3126
2084
1389
926
"

MOSES_PATH=/fs/lofn0/dmadl/software/mosesdecoder; MOSES=$MOSES_PATH; export PATH=$MOSES_PATH/bin:$MOSES_PATH/scripts:$PATH; WD=$(pwd);

mkdir -p train train.bin
for ntrain in $SIZES; do
    head -n $ntrain ../bigcorpus.perm.en.head > train/corpus.$ntrain
    $MOSES/bin/lmplz -o 5 -S 15% -T /tmp < train/corpus.$ntrain > train/lm.$ntrain
    $MOSES/bin/build_binary train/lm.$ntrain train.bin/$ntrain
done
