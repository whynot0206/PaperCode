# -*- encoding:utf-8 -*-
import os
import torch
from multiprocessing import Pool
from uer.utils.constants import *
from uer.utils.misc import count_lines


class Vocab(object):
    """
    """

    # 初始化词汇表对象，创建三个核心数据结构：w2i（词到索引的字典）、
    # i2w（索引到词的列表）和 w2c（词到计数的字典）。同时设置保留词汇表文件的路径，该文件包含预定义的特殊符号。
    def __init__(self):
        self.w2i = {}
        self.i2w = []
        self.w2c = {}
        self.reserved_vocab_path = \
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/reserved_vocab.txt"))

    # 从指定路径加载已存在的词汇表文件。读取每行作为一个词，建立词到索引和索引到词的双向映射。可选择是否安静模式（不打印词汇表大小）。
    def load(self, vocab_path, is_quiet=False):
        with open(vocab_path, mode="r", encoding="utf-8") as reader:
            for index, line in enumerate(reader):
                w = line.strip("\n").split()[0] if line.strip() else line.strip("\n")
                self.w2i[w] = index
                self.i2w.append(w)
        if not is_quiet:
            print("Vocabulary size: ", len(self))

    # 将当前词汇表保存到指定文件。每个词占一行，按索引顺序写入，并输出词汇表大小和保存完成信息。
    def save(self, save_path):
        print("Vocabulary size: ", len(self))
        with open(save_path, mode="w", encoding="utf-8") as f:
            for w in self.i2w:
                f.write(w + "\n")
        print("Vocabulary saving done.")

    # 根据词获取对应的索引值。是 w2i 字典的便捷封装。
    def get(self, w):
        return self.w2i[w]

    # 返回词汇表的大小（即包含的词数量）。
    def __len__(self):
        return len(self.i2w)

    # 多进程工作函数，处理语料库的指定范围（从 start 到 end 行）。
    # 使用分词器对每行文本进行分词，统计每个词的出现频率，并构建局部词汇表。
    # 返回该进程范围内的词汇统计数据。
    def worker(self, corpus_path, tokenizer, start, end):
        """ 
        Worker that creates vocabulary from corpus[start:end].
        """
        w2i, i2w, w2c = {}, [], {}
        pos = 0
        with open(corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                tokens = tokenizer.tokenize(line, use_vocab=False)
                for t in tokens:
                    if t not in w2i:
                        w2i[t], w2c[t] = len(i2w), 1
                        i2w.append(t)
                    else:
                        w2c[t] += 1
                if pos >= end - 1:
                    return (w2i, i2w, w2c)

    # 合并多个工作进程生成的词汇表。将所有进程中的词合并到一个统一的词汇表中，并累加相同词的计数。
    def union(self, vocab_list):
        """ Union vocab in all workers. """
        w2i, i2w, w2c = {}, [], {}
        index = 0
        for v_p in vocab_list:
            w2i_p, i2w_p, w2c_p = v_p.get()
            for w in i2w_p:
                if w not in w2i:
                    w2i[w], w2c[w] = len(i2w), w2c_p[w]
                    i2w.append(w)
                else:
                    w2c[w] += w2c_p[w]
        return (w2i, i2w, w2c)

    # 从给定语料库构建完整的词汇表。使用多进程并行处理语料库，每个进程处理一部分数据，然后合并所有结果。
    # 首先加载保留的特殊符号，然后按词频从高到低添加语料中的词，过滤掉出现次数低于 min_count 的低频词。
    def build(self, corpus_path, tokenizer, workers_num=1, min_count=1):
        """ Build vocabulary from the given corpus. """
        print("Start %d workers for building vocabulary..." % workers_num)
        lines_num = count_lines(corpus_path)
        pool = Pool(workers_num)
        vocab_list = []
        for i in range(workers_num):
            start = i * lines_num // workers_num
            end = (i + 1) * lines_num // workers_num
            vocab_list.append((pool.apply_async(func=self.worker, args=[corpus_path, tokenizer, start, end])))
        pool.close()
        pool.join()

        # Union vocab in all workers.
        w2i, i2w, w2c = self.union(vocab_list)
        # Sort w2c according to word count.
        sorted_w2c = sorted(w2c.items(), key=lambda item: item[1], reverse=True)

        # Add special symbols and remove low frequency words.
        with open(self.reserved_vocab_path, mode="r", encoding="utf-8") as reader:
            self.i2w = [line.strip().split()[0] for line in reader]

        for i, w in enumerate(self.i2w):
            self.w2i[w] = i
            self.w2c[w] = -1

        for w, c in sorted_w2c:
            if c < min_count:
                break
            if w not in self.w2i:
                self.w2i[w], self.w2c[w] = len(self.i2w), c
                self.i2w.append(w)
