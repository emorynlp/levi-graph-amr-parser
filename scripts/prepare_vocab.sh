dataset=$1
python3 -u -m amr_parser.extract --train_data ${dataset}/train.txt.features.preproc --levi_graph $2
rm -f ${dataset}/*_vocab
mv *_vocab ${dataset}/
# python3 encoder.py
# cat ${dataset}/*embed | sort | uniq > ${dataset}/glove.embed.txt
# rm ${dataset}/*embed
