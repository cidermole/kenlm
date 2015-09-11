#!/bin/bash

#MOSES_PATH=/fs/lofn0/dmadl/software/mosesdecoder; MOSES=$MOSES_PATH; export PATH=$MOSES_PATH/bin:$MOSES_PATH/scripts:$PATH; WD=$(pwd);
#ntest=200000

KENLM=/fs/syn0/dmadl/kenlm

test=bigcorpus.perm.en.tail

for model in train.bin/*; do
    echo -n $(basename $model)

    # echo "Building vocab..." >&2
    $KENLM/bin/kenlm_benchmark vocab $model < "$test" > /tmp/lm.vocab 2>/dev/null
    
    #Ensure files are in RAM.
    cat /tmp/lm.vocab $model >/dev/null
    
    # no prefetch (old)
    #./test.sh $model 2>/dev/null | awk '/total_runtime/ { print " " $3 }'
    $KENLM/bin/kenlm_benchmark.old query $model < /tmp/lm.vocab 2>/dev/null | awk '/total_runtime/ { print " " $3 }'
    for nprefetch in 1 2 5 10; do
        # various prefetch models
        #./test.sh $model $nprefetch 2>/dev/null | awk '/total_runtime/ { print " " $3 }'
        $KENLM/bin/kenlm_benchmark query $model $nprefetch < /tmp/lm.vocab 2>/dev/null | awk '/total_runtime/ { print " " $3 }'
    done
done
