from vocab_gen import build_BPE, build_vocab

merged_corpus_file = "data/pretrain/corpus_temp_biburst.txt"
vocab_file = "../models/encryptd_vocab.txt"

print("开始执行步骤3：构建词表...")

build_BPE(merged_corpus_file)
build_vocab(vocab_file)

print("步骤3完成！词表已输出:", vocab_file)
