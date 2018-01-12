th preprocess.lua -train_src ../convai-bot-1337/fbnews_chitchat/posts.train.txt \
  -train_tgt ../convai-bot-1337/fbnews_chitchat/comments.train.txt \
  -valid_src ../convai-bot-1337/fbnews_chitchat/posts.test.txt \
  -valid_tgt ../convai-bot-1337/fbnews_chitchat/comments.test.txt \
  -save_data ../convai-bot-1337/fbnews_chitchat/data/opennmt/dataset \
  -data_type bitext \
  -src_vocab_size 100000 \
  -tgt_vocab_size 100000
