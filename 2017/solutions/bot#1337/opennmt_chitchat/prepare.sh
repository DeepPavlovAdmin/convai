th tools/build_vocab.lua -data ../convai/data/opennmt/open_sub_ver3.shuf.train.src \
-save_vocab ../convai/data/opennmt/vocab_ver4

# split для разделения на n файлов


th preprocess.lua -train_src ../convai/data/opennmt/open_sub_ver3.shuf.train.src \
  -train_tgt ../convai/data/opennmt/open_sub_ver3.shuf.train.tgt \
  -valid_src ../convai/data/opennmt/open_sub_ver3.shuf.test.src \
  -valid_tgt ../convai/data/opennmt/open_sub_ver3.shuf.test.tgt \
  -save_data ../convai/data/opennmt/dataset \
  -data_type bitext \
  -src_vocab_size 50000 \
  -tgt_vocab_size 50000 \
  -src_seq_length 20 \
  -tgt_seq_length 20
