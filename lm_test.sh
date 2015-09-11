#!/bin/bash

#MOSES_PATH=/fs/lofn0/dmadl/software/mosesdecoder; MOSES=$MOSES_PATH; export PATH=$MOSES_PATH/bin:$MOSES_PATH/scripts:$PATH; WD=$(pwd);
#ntest=200000
# about 3M words test: 
# dmadl@syn:/fs/syn0/dmadl/bigcorpus$ wc bigcorpus.perm.en.tail 
#  200000  3207891 17679412 bigcorpus.perm.en.tail

KENLM=/fs/sif0/dmadl/kenlm

test=test.txt
lookups=$(wc $test | awk '{ print $2 }')

divide() {
    echo $1 $2 | python -c 'from __future__ import division, print_function; import sys; nums = [float(n) for n in sys.stdin.readline().split(" ")]; print("%.4e" % (nums[0] / nums[1]))'
}

mkdir -p test
for model in train.bin/*; do
    nlines_train=$(basename $model)
    mkdir -p test/$nlines_train
    #echo -n $nlines_train
    
    # actually, about 100 less?? (should be ~18 less)
    model_size=$(wc -l train/lm.$nlines_train | awk '{print $1}')
    echo -n $model_size

    # echo "Building vocab..." >&2
    $KENLM/bin/kenlm_benchmark vocab $model < "$test" > /tmp/lm.vocab 2>/dev/null
    
    #Ensure files are in RAM.
    cat /tmp/lm.vocab $model >/dev/null
    
    # no prefetch (old)
    #./test.sh $model 2>/dev/null | awk '/total_runtime/ { print " " $3 }'
    per_lookup=$($KENLM/bin/kenlm_benchmark.old query $model < /tmp/lm.vocab 2>test/$nlines_train/old.err | tee test/$nlines_train/old.out | awk '/CPU_per_query/ { print $4 }')
    echo -n " $per_lookup"
    for nprefetch in 1 2 5 10; do
        # various prefetch models
        #./test.sh $model $nprefetch 2>/dev/null | awk '/total_runtime/ { print " " $3 }'
        per_lookup=$($KENLM/bin/kenlm_benchmark query $model $nprefetch < /tmp/lm.vocab 2>test/$nlines_train/$nprefetch.err | tee test/$nlines_train/$nprefetch.out | awk '/CPU_per_query/ { print $4 }')
        echo -n " $per_lookup"
    done
    echo ""
done
