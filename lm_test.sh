#!/bin/bash

MOSES_PATH=/fs/lofn0/dmadl/software/mosesdecoder; MOSES=$MOSES_PATH; export PATH=$MOSES_PATH/bin:$MOSES_PATH/scripts:$PATH; WD=$(pwd);
#ntest=200000

for nprefetch in 1 2 5 10; do
    for size in train.bin/*; do
        #./test.sh $size $nprefetch 2>&1 >/dev/null | 
        ./test.sh $size $nprefetch 2>/dev/null | awk '/total_runtime/ { print $3 }'
    done
done
