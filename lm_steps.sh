#!/bin/bash

NTEST=200000
STEPS=15

nlines=177776463
MAX=$(expr $nlines - $NTEST)
MIN=10

SIZES=$(python -c "
from __future__ import division
max=$MAX
min=$MIN
steps=$STEPS
fac=(max/min)**(1/(steps-1))
i=max
while i >= min-1:
    print(int(i))
    i /= fac
")

echo "$SIZES" > sizes.txt

# about
# 100 B - 100 GB language models
# cat_compressed

MOSES_PATH=/fs/lofn0/dmadl/software/mosesdecoder; MOSES=$MOSES_PATH; export PATH=$MOSES_PATH/bin:$MOSES_PATH/scripts:$PATH; WD=$(pwd);

mkdir -p train train.bin
for ntrain in $SIZES; do
    head -n $ntrain gigaword_uncompressed.txt > train/corpus.$ntrain
    $MOSES/bin/lmplz -o 5 -S 80% -T /tmp < train/corpus.$ntrain > train/lm.$ntrain
    $MOSES/bin/build_binary train/lm.$ntrain train.bin/$ntrain
done

tail -n $NTEST gigaword_uncompressed.txt > test.txt
