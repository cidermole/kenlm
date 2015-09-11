#!/bin/bash

LM_BIN=/home/david/tmp/mt/wd/corpus/lm/emea.train.en.bin

if [ ! -f $LM_BIN ]; then
	bin/build_binary /home/david/tmp/mt/wd/corpus/lm/emea.train.en $LM_BIN
fi
echo "Building vocab..." >&2
bin/kenlm_benchmark vocab $LM_BIN < "$1" > /tmp/lm.vocab 2>/dev/null
echo "Running benchmark..." >&2
time bin/kenlm_benchmark query $LM_BIN "$2" < /tmp/lm.vocab 2>/dev/null
