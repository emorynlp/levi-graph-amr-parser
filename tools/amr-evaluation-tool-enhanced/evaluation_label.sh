#!/bin/bash

# Evaluation script. Run as: ./evaluation.sh <parsed_data> <gold_data>
p=$$_pred.tmp
g=$$_gold.tmp
python unlabel.py "$1" >$p
python unlabel.py "$2" >$g

out=$(/usr/bin/python2 smatch/smatch.py --pr -f $p $g)
out=($out)
echo 'Unlabeled -> P: '${out[1]}', R: '${out[3]}', F: '${out[6]} | sed 's/.$//'

rm $p
rm $g
