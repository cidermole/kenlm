#!/bin/bash

NTEST=200000
STEPS=15

nlines=177776463
MAX=$(expr $nlines - $NTEST)
MIN=2

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
i=0
for ntrain in $SIZES; do
    (head -n $ntrain gigaword_uncompressed.txt > train/corpus.$ntrain;
    $MOSES/bin/lmplz -o 5 -S 20% -T /tmp < train/corpus.$ntrain > train/lm.$ntrain;
    $MOSES/bin/build_binary train/lm.$ntrain train.bin/$ntrain;
    ) &
    i=$(expr $i + 1)
    if [ $i -eq 4 -o $i -eq 8 -o $i -eq 12 ]; then
        wait
    fi
done
wait

tail -n $NTEST gigaword_uncompressed.txt > test.txt
