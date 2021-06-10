#!/bin/bash

# Evaluation script. Run as: ./evaluation.sh <parsed_data> <gold_data>
out=`/usr/bin/python2 smatch/smatch.py --pr -f "$1" "$2"`
out=($out)
echo 'Smatch -> P: '${out[1]}', R: '${out[3]}', F: '${out[6]} | sed 's/.$//'
p=$$_pred.tmp
g=$$_gold.tmp
python unlabel.py "$1"  > $p
python unlabel.py "$2"  > $g

out=`/usr/bin/python2 smatch/smatch.py --pr -f $p $g`
out=($out)
echo 'Unlabeled -> P: '${out[1]}', R: '${out[3]}', F: '${out[6]} | sed 's/.$//'

cat "$1" | perl -ne 's/(\/ [a-zA-Z0-9\-][a-zA-Z0-9\-]*)-[0-9][0-9]*/\1-01/g; print;' > $p
cat "$2" | perl -ne 's/(\/ [a-zA-Z0-9\-][a-zA-Z0-9\-]*)-[0-9][0-9]*/\1-01/g; print;' > $g
out=`/usr/bin/python2 smatch/smatch.py --pr -f $p $g`
out=($out)
echo 'No WSD -> P: '${out[1]}', R: '${out[3]}', F: '${out[6]} | sed 's/.$//'

cat "$1" | perl -ne 's/^#.*\n//g; print;' | tr '\t' ' ' | tr -s ' ' > $p
cat "$2" | perl -ne 's/^#.*\n//g; print;' | tr '\t' ' ' | tr -s ' ' > $g
/usr/bin/python2 scores.py "$p" "$g"

rm $p
rm $g
