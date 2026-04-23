from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors  # 从tokenizers导入相关模块
import json  # 导入JSON数据处理模块
import os  # 导入操作系统接口模块

'''def _corpus_iterator(corpora_path):
    with open(corpora_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def build_BPE(corpora_path):
    # generate source dictionary,0-65535 
    num_count = 65536
    not_change_string_count = 5
    i = 0
    source_dictionary = {} 
    tuple_sep = ()
    tuple_cls = ()
    #'PAD':0,'UNK':1,'CLS':2,'SEP':3,'MASK':4
    while i < num_count:
        temp_string = '{:04x}'.format(i) 
        source_dictionary[temp_string] = i
        i += 1
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.WordPiece(vocab=source_dictionary,unk_token="[UNK]",max_input_chars_per_word=4))

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.post_processor = processors.BertProcessing(sep=("[SEP]",1),cls=('[CLS]',2))

    # And then train
    trainer = trainers.WordPieceTrainer(vocab_size=65536, min_frequency=2)
    tokenizer.train_from_iterator(_corpus_iterator(corpora_path), trainer=trainer)

    # And Save it
    tokenizer.save("wordpiece.tokenizer.json", pretty=True)
    return 0
'''


def build_BPE(corpora_path):  # 构建BPE分词器的函数
    # generate source dictionary,0-65535   # 生成源字典，0-65535
    num_count = 65536  # 设置字典大小
    not_change_string_count = 5  # 设置不变字符串数量
    i = 0  # 初始化计数器
    source_dictionary = {}  # 初始化源字典
    tuple_sep = ()  # 初始化分隔符元组（未使用）
    tuple_cls = ()  # 初始化CLS元组（未使用）
    # 'PAD':0,'UNK':1,'CLS':2,'SEP':3,'MASK':4  # 特殊token的映射
    while i < num_count:  # 循环生成字典
        temp_string = '{:04x}'.format(i)  # 将数字格式化为4位十六进制字符串
        source_dictionary[temp_string] = i  # 添加到源字典
        i += 1  # 计数器加1
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())  # 改为使用BPE模型，更节省内存

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # 改为ByteLevel预分词器
    tokenizer.decoder = decoders.ByteLevel()  # 改为ByteLevel解码器
    tokenizer.post_processor = processors.BertProcessing(sep=("[SEP]", 1), cls=('[CLS]', 2))  # 设置后处理器为BERT后处理器

    # And then train
    trainer = trainers.BpeTrainer(  # 改为BPE训练器
        vocab_size=65536,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # 添加初始字母表
    )

    # 修改后：只传一次训练文件路径
    tokenizer.train([corpora_path], trainer=trainer)

    # And Save it
    tokenizer.save("wordpiece.tokenizer.json", pretty=True)  # 保存分词器到JSON文件
    return 0  # 返回成功代码


def build_vocab(vocab_path):  # 构建词汇表的函数
    json_file = open("wordpiece.tokenizer.json",'r')  # 打开分词器JSON文件
    json_content = json_file.read()  # 读取文件内容
    json_file.close()  # 关闭文件
    vocab_json = json.loads(json_content)  # 解析JSON内容
    vocab_txt = ["[PAD]","[SEP]","[CLS]","[UNK]","[MASK]"]  # 初始化词汇表（特殊token）
    for item in vocab_json['model']['vocab']:  # 遍历分词器词汇表
        vocab_txt.append(item) # append key of vocab_json  # 添加词汇表项
    with open(vocab_path,'w') as f:  # 打开词汇表文件
        for word in vocab_txt:  # 遍历词汇表
            f.write(word+"\n")  # 写入词汇表项（每行一个）
    return 0  # 返回成功代码
