import os  # 导入操作系统接口模块，用于文件和目录操作
import random  # 导入随机数生成模块，用于数据随机化
import pickle  # 导入序列化模块，用于将Python对象保存到文件
import torch  # 导入PyTorch深度学习框架
from multiprocessing import Pool  # 导入多进程池，用于并行处理数据
from uer.utils.constants import *  # 从UER工具包导入所有常量
from uer.utils.tokenizers import *  # 从UER工具包导入所有分词器
from uer.utils.misc import count_lines  # 导入统计文件行数的函数
from uer.utils.seed import set_seed  # 导入设置随机种子的函数
import os
import pickle

# 该函数用于对输入序列进行掩码处理，支持三种掩码策略：
# 单词级掩码、片段掩码和普通掩码。它会随机选择一定比例的 token 进行替换（替换为 [MASK]、随机词或保持不变），
# 并返回掩码后的序列和掩码位置及原始 token 的列表。
def mask_seq(src, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length):
    vocab = tokenizer.vocab

    for i in range(len(src) - 1, -1, -1):
        if src[i] != PAD_ID:
            break
    src_no_pad = src[:i + 1]
    tokens_index, src_no_pad = create_index(src_no_pad, tokenizer, whole_word_masking, span_masking, span_geo_prob,
                                            span_max_length)
    if len(src_no_pad) < len(src):
        src = src_no_pad + (len(src) - len(src_no_pad)) * [PAD_ID]
    else:
        src = src_no_pad

    random.shuffle(tokens_index)
    num_to_predict = max(1, int(round(len(src_no_pad) * 0.15)))
    tgt_mlm = []
    for index_set in tokens_index:
        if len(tgt_mlm) >= num_to_predict:
            break
        if whole_word_masking:
            i = index_set[0]
            mask_len = index_set[1]
            if len(tgt_mlm) + mask_len > num_to_predict:
                continue

            for j in range(mask_len):
                token = src[i + j]
                tgt_mlm.append((i + j, token))
                prob = random.random()
                if prob < 0.8:
                    src[i + j] = vocab.get(MASK_TOKEN)
                elif prob < 0.9:
                    while True:
                        rdi = random.randint(1, len(vocab) - 1)
                        if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                            break
                    src[i + j] = rdi
        elif span_masking:
            i = index_set[0]
            span_len = index_set[1]
            if len(tgt_mlm) + span_len > num_to_predict:
                continue

            for j in range(span_len):
                token = src[i + j]
                tgt_mlm.append((i + j, token))
            prob = random.random()
            if prob < 0.8:
                for j in range(span_len):
                    src[i + j] = vocab.get(MASK_TOKEN)
            elif prob < 0.9:
                for j in range(span_len):
                    while True:
                        rdi = random.randint(1, len(vocab) - 1)
                        if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                            break
                    src[i + j] = rdi
        else:
            i = index_set[0]
            token = src[i]
            tgt_mlm.append((i, token))
            prob = random.random()
            if prob < 0.8:
                src[i] = vocab.get(MASK_TOKEN)
            elif prob < 0.9:
                while True:
                    rdi = random.randint(1, len(vocab) - 1)
                    if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                        break
                src[i] = rdi
    tgt_mlm = sorted(tgt_mlm, key=lambda x: x[0])
    return src, tgt_mlm


# 根据是否启用整词掩码或片段掩码，生成序列中需要掩码的位置索引。如果是整词掩码，
# 会使用 jieba 分词将序列划分为词，记录每个词的起始位置；
# 如果是片段掩码，则根据几何分布生成片段长度并记录起始位置。
def create_index(src, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length):
    tokens_index = []
    span_end_position = -1
    vocab = tokenizer.vocab
    if whole_word_masking:
        src_wwm = []
        src_length = len(src)
        has_cls, has_sep = False, False
        if src[0] == vocab.get(CLS_TOKEN):
            src = src[1:]
            has_cls = True
        if src[-1] == vocab.get(SEP_TOKEN):
            src = src[:-1]
            has_sep = True
        sentence = "".join(tokenizer.convert_ids_to_tokens(src)).replace('[UNK]', '').replace('##', '')
        import jieba
        wordlist = jieba.cut(sentence)
        if has_cls:
            src_wwm += [vocab.get(CLS_TOKEN)]
        for word in wordlist:
            position = len(src_wwm)
            src_wwm += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            if len(src_wwm) < src_length:
                tokens_index.append([position, len(src_wwm) - position])
        if has_sep:
            src_wwm += [vocab.get(SEP_TOKEN)]
        if len(src_wwm) > src_length:
            src = src_wwm[:src_length]
        else:
            src = src_wwm
    else:
        for (i, token) in enumerate(src):
            if token == vocab.get(CLS_TOKEN) or token == vocab.get(SEP_TOKEN) or token == PAD_ID:
                continue
            if not span_masking:
                tokens_index.append([i])
            else:
                if i < span_end_position:
                    continue
                span_len = get_span_len(span_max_length, span_geo_prob)
                span_end_position = i + span_len
                if span_end_position > len(src):
                    span_len = len(src) - i
                tokens_index.append([i, span_len])
    return tokens_index, src


# 根据几何分布概率生成一个片段长度，用于片段掩码。通过累积概率分布随机选择一个长度值，控制掩码片段的长度。
def get_span_len(max_span_len, p):
    geo_prob_cum = [0.0]
    geo_prob = 1.0
    for i in range(max_span_len + 1):
        if i == 0:
            continue
        if i == 1:
            geo_prob *= p
            geo_prob_cum.append(geo_prob_cum[-1] + geo_prob)
        else:
            geo_prob *= (1 - p)
            geo_prob_cum.append(geo_prob_cum[-1] + geo_prob)

    prob = geo_prob_cum[-1] * random.random()
    for i in range(len(geo_prob_cum) - 1):
        if prob >= geo_prob_cum[i] and prob < geo_prob_cum[i + 1]:
            current_span_len = i + 1
    return current_span_len


# 将多个进程生成的临时数据集文件合并为一个完整的数据集文件。每个进程生成的文件被读取并写入到最终输出文件中，然后删除临时文件。
'''def merge_dataset(dataset_path, workers_num):
    # Merge datasets.
    dataset_writer = open(dataset_path, "wb")
    for i in range(workers_num):
        tmp_dataset_reader = open("/mnt/data/zgm/ET-BERT/datasets/temp/dataset-tmp-" + str(i) + ".pt", "rb")
        while True:
            tmp_data = tmp_dataset_reader.read(2 ^ 20)
            if tmp_data:
                dataset_writer.write(tmp_data)
            else:
                break
        tmp_dataset_reader.close()
        os.remove("/mnt/data/zgm/ET-BERT/datasets/temp/dataset-tmp-" + str(i) + ".pt")
    dataset_writer.close()
'''


def merge_dataset(dataset_path, workers_num):
    # --- 修改开始 ---
    # 强制转换为绝对路径，避免相对路径歧义
    abs_dataset_path = os.path.abspath(dataset_path)

    # 基于绝对路径构建临时目录
    tmp_dir = os.path.join(os.path.dirname(abs_dataset_path), "temp_datasets")

    if not os.path.exists(tmp_dir):
        print(f"Notice: Temporary directory {tmp_dir} was not found. Creating it now.")
        os.makedirs(tmp_dir, exist_ok=True)  # 使用 exist_ok=True 防止并发错误

    # Merge datasets.
    dataset_writer = open(abs_dataset_path, "wb")
    print(f"Merging temporary files into final dataset at: {abs_dataset_path}")

    for i in range(workers_num):
        temp_file_path = os.path.join(tmp_dir, "dataset-tmp-" + str(i) + ".pt")

        if not os.path.exists(temp_file_path):
            print(f"Warning: Temporary file not found: {temp_file_path}. Skipping this worker's output.")
            continue

        print(f"Reading temporary file: {temp_file_path}")

        try:
            tmp_dataset_reader = open(temp_file_path, "rb")

            # 使用更安全的读取方式，例如逐字节读取或直接加载
            # 原始代码使用了 2^20，我们继续使用这个逻辑，但为了安全起见，使用 math.pow(2, 20) 或 1024 * 1024
            chunk_size = 1024 * 1024  # 1MB

            while True:
                tmp_data = tmp_dataset_reader.read(chunk_size)
                if tmp_data:
                    dataset_writer.write(tmp_data)
                else:
                    break
            tmp_dataset_reader.close()

            # 删除临时文件
            os.remove(temp_file_path)

        except Exception as e:
            print(f"An error occurred while merging file {temp_file_path}: {e}")

    dataset_writer.close()
    print("Dataset merging complete.")

# 对两个序列进行截断，使它们的总长度不超过指定值。随机从较长序列中删除 token，直到满足长度要求。
def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """ truncate sequence pair to specific length """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


# 记录语料库中每个流（以“||”开头的行）的起始行号，返回这些起始位置的列表，用于后续按流分割数据。
def record_flow_start(corpus_path):
    starts = []
    with open(corpus_path, mode="r", encoding="utf-8") as f:
        i = 0
        while True:
            line = f.readline()
            if not line:
                break
            if line[:2] == "||":
                starts.append(i)
            i += 1
    starts.append(i)
    return starts


# 数据集处理的基础类，初始化参数包括词汇表、分词器、语料路径、序列长度等。定义了构建和保存数据集的基本结构，支持多进程处理。
class Dataset(object):
    def __init__(self, args, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.corpus_path = args.corpus_path
        self.dataset_path = args.dataset_path
        self.seq_length = args.seq_length
        self.seed = args.seed
        self.dynamic_masking = args.dynamic_masking
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length
        self.docs_buffer_size = args.docs_buffer_size
        self.dup_factor = args.dup_factor

    def build_and_save(self, workers_num, split_by_flow=False):
        """
        Build dataset from the given corpus.
        Start workers_num processes and each process deals with a part of data.
        """
        lines_num = count_lines(self.corpus_path)
        print("Starting %d workers for building datasets ... " % workers_num)
        assert (workers_num >= 1)
        if workers_num == 1:
            self.worker(0, 0, lines_num)
        else:
            pool = Pool(workers_num)
            if split_by_flow:
                starts = record_flow_start(self.corpus_path)
            current_index = 0
            for i in range(workers_num):
                if split_by_flow:
                    # start = starts[current_index]
                    # for j in range(len(starts))[current_index:]:
                    #     if starts[j]-starts[current_index]>perburst:
                    #         current_index = j
                    #         break
                    # if i==workers_num-1:
                    #     current_index = len(starts)-1
                    # end = starts[current_index]
                    start = starts[i * (len(starts) - 1) // workers_num]
                    end = starts[(i + 1) * (len(starts) - 1) // workers_num]
                else:
                    start = i * lines_num // workers_num
                    end = (i + 1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])
            pool.close()
            pool.join()

        # Merge datasets.
        merge_dataset(self.dataset_path, workers_num)

    def worker(self, proc_id, start, end):
        raise NotImplementedError()


# 数据加载器的基础类，负责从数据集中读取数据并分批返回。支持多进程读取、缓冲区管理和随机打乱。
class DataLoader(object):
    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=False):
        self.tokenizer = args.tokenizer
        self.batch_size = batch_size
        self.instances_buffer_size = args.instances_buffer_size
        self.proc_id = proc_id
        self.proc_num = proc_num
        self.shuffle = shuffle
        self.dataset_reader = open(dataset_path, "rb")
        self.read_count = 0
        self.start = 0
        self.end = 0
        self.buffer = []
        self.vocab = args.vocab
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length

    def _fill_buf(self):
        try:
            self.buffer = []
            while True:
                instance = pickle.load(self.dataset_reader)
                self.read_count += 1
                if (self.read_count - 1) % self.proc_num == self.proc_id:
                    self.buffer.append(instance)
                    if len(self.buffer) >= self.instances_buffer_size:
                        break
        except EOFError:
            # Reach file end.
            self.dataset_reader.seek(0)

        if self.shuffle:
            random.shuffle(self.buffer)
        self.start = 0
        self.end = len(self.buffer)

    def _empty(self):
        return self.start >= self.end

    def __del__(self):
        self.dataset_reader.close()


# 继承自 Dataset，用于构建 BERT 预训练任务（MLM 和 NSP）的数据集。从语料中读取文档，生成句子对，并应用动态或静态掩码。
class BertDataset(Dataset):
    """
    Construct dataset for MLM and NSP tasks from the given corpus.
    Each document consists of multiple sentences,
    and each sentence occupies a single line.
    Documents in corpus must be separated by empty lines.
    从给定语料库构建MLM和NSP任务的数据集。
    每个文档由多个句子组成，
    每个句子占据单独一行。
    语料库中的文档必须用空行分隔。
    """

    def __init__(self, args, vocab, tokenizer):
        super(BertDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        docs_buffer = []
        document = []
        pos = 0
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.path.dirname(self.dataset_path), "temp_datasets")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        dataset_writer = open(os.path.join(temp_dir, "dataset-tmp-" + str(proc_id) + ".pt"), "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1
                if pos >= end:
                    if len(docs_buffer) > 0:
                        instances = self.build_instances(docs_buffer)
                        for instance in instances:
                            pickle.dump(instance, dataset_writer)
                    break

                if not line.strip():
                    if len(document) >= 1:
                        docs_buffer.append(document)
                    document = []
                    if len(docs_buffer) == self.docs_buffer_size:
                        print("Worker {}: Docs buffer is full, build instances...".format(proc_id))
                        # Build instances from documents.
                        instances = self.build_instances(docs_buffer)
                        # Save instances.
                        for instance in instances:
                            pickle.dump(instance, dataset_writer)
                        # Clear buffer.
                        docs_buffer = []
                    continue
                if line[:2] == "||":
                    line = line[2:]
                sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                if len(sentence) > 0:
                    document.append(sentence)

        dataset_writer.close()

    def build_instances(self, all_documents):
        instances = []
        for _ in range(self.dup_factor):
            for doc_index in range(len(all_documents)):
                instances.extend(self.create_ins_from_doc(all_documents, doc_index))
        return instances

    def create_ins_from_doc(self, all_documents, document_index):
        document = all_documents[document_index]
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []  # seg 1
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []  ##seg 2
                    is_random_next = 0

                    if len(current_chunk) == 1 or random.random() < 0.5:
                        is_random_next = 1
                        target_b_length = target_seq_length - len(tokens_a)

                        for _ in range(10):
                            random_document_index = random.randint(0, len(all_documents) - 1)
                            if random_document_index != document_index:
                                break

                        random_document = all_documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break

                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments

                    else:
                        is_random_next = 0
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))

                    while len(src) != self.seq_length:
                        src.append(PAD_ID)

                    if not self.dynamic_masking:
                        src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking,
                                                self.span_geo_prob, self.span_max_length)
                        instance = (src, tgt_mlm, is_random_next, seg_pos)
                    else:
                        instance = (src, is_random_next, seg_pos)

                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances


# 扩展自 Dataset，专门用于处理流式数据（如网络流量），支持 MLM 和 MIX 任务。按流（flow）组织数据，并生成带有协议标签的实例。
class BertFlowDataset(Dataset):
    """
    Construct dataset for MLM and MIX tasks from the given corpus.
    Each document consists of multiple paragraphs,
    Each paragraph consists of multiple sentences,
    and each sentence occupies a single line.
    Paragraphs in corpus must be separated by empty lines.
    Documents in corpus must be separated by empty lines.
    为MLM和MIX任务从给定语料库构建数据集。
    每个文档由多个段落组成，
    每个段落由多个句子组成，
    每个句子占据单独一行。
    语料库中的段落必须用空行分隔。
    语料库中的文档必须用空行分隔。
    """

    def __init__(self, args, vocab, tokenizer):
        super(BertFlowDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        print(start, end)
        set_seed(self.seed)
        flow_buffer = []
        flow_proto = []
        docs_buffer = []
        document = []
        pos = 0
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.path.dirname(self.dataset_path), "temp_datasets")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        dataset_writer = open(os.path.join(temp_dir, "dataset-tmp-" + str(proc_id) + ".pt"), "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            try:
                # with open(self.corpus_path[:-4]+"_extra.txt", mode="r", encoding="utf-8") as fe:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    if pos == start and line[:2] != "||":
                        print("not flow start...")
                    pos += 1
                    if pos > end:
                        if len(docs_buffer) >= 1:
                            flow_buffer.append(docs_buffer)

                        if len(flow_buffer) > 0:
                            try:
                                instances = self.build_instances(flow_buffer, flow_proto)
                            except Exception as e:
                                print("has error2...", len(flow_buffer), len(flow_proto), e)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        break

                    if not line.strip():
                        if len(document) >= 1:
                            docs_buffer.append(document)
                        document = []
                        continue
                    if line[:2] == "||" or not line:
                        if len(docs_buffer) >= 1:
                            flow_buffer.append(docs_buffer)
                        docs_buffer = []

                        flow_buffer_size = 0
                        for d in flow_buffer:
                            flow_buffer_size += len(d)
                        if flow_buffer_size > self.docs_buffer_size or pos >= end:
                            print("Worker %d is building instances ... " % proc_id, len(flow_buffer), len(flow_proto))
                            try:
                                instances = self.build_instances(flow_buffer, flow_proto)
                            except:
                                print("has error1...")
                            print("Worker {} has {} instances. ".format(proc_id, len(instances)))
                            # Save instances.
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                            flow_buffer = []
                            flow_proto = []
                        if pos >= end or not line:
                            break
                        line = line[2:]
                        if line[0] == "4":
                            if "bigram" in self.corpus_path:
                                if line[42:44] == "06":
                                    flow_proto.append(0)
                                elif line[42:44] == "11":
                                    flow_proto.append(1)
                                else:
                                    print("not tcp or udp, ", line[42:44])
                            else:
                                if line[22:24] == "06":
                                    flow_proto.append(0)
                                elif line[22:24] == "11":
                                    flow_proto.append(1)
                        else:
                            print("find Ipv6!!")

                    sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                    if len(sentence) > 0:
                        document.append(sentence)

            except:
                print("has error...")
        print("Worker %d finished... " % proc_id)
        dataset_writer.close()

    def build_instances(self, all_documents, flow_proto):
        assert len(all_documents) == len(flow_proto)
        instances = []
        for _ in range(self.dup_factor):
            for doc_index in range(len(all_documents)):
                # if doc_index%500==0:
                #     print(doc_index)
                instances.extend(self.create_ins_from_doc(all_documents, doc_index, flow_proto))
        return instances

    def create_ins_from_doc(self, all_documents, document_index, flow_proto):
        # all_documents: 所有流量数据的列表，每个元素是一个 Flow（包含多个 Bursts）
        # document_index: 当前处理的 Flow 在列表中的索引
        # flow_proto: 对应的协议类型列表（用于 MoE 机制，如 TCP=0, UDP=1）

        document = all_documents[document_index]  # 获取当前要处理的 Flow（由多个 Burst 组成的列表）
        max_num_tokens = self.seq_length - 3  # 计算最大有效 Token 数，减 3 是为了留给 [CLS], [SEP], [SEP] 这三个特殊 Token
        target_seq_length = max_num_tokens  # 设定目标序列长度，默认为最大长度

        # ------------------------------------------------------------------
        # 策略：短序列采样
        # 为了让模型也能适应较短的输入，以一定概率（short_seq_prob）随机生成一个较短的目标长度
        # ------------------------------------------------------------------
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)

        instances = []  # 用于存放生成的训练样本
        i = 0  # 初始化 Burst 的遍历索引

        # 开始遍历当前 Flow 中的每一个 Burst
        while i < len(document):
            rnd = random.random()  # 生成一个 0-1 之间的随机数，决定当前 Burst 用于构建哪种任务

            # ==============================================================================
            # 分支 1：单 Burst 切分任务 (对应 SODF Label 0 & 1)
            # 触发条件：如果是最后一个 Burst，或者随机数 < 0.4 (40% 概率)
            # ==============================================================================
            if i == len(document) - 1 or rnd < 0.4:
                a_end = 1  # 初始化切分点
                if len(document[i]) >= 2:
                    a_end = random.randint(1, len(document[i]) - 1)  # 随机选择当前 Burst 内部的一个切分点

                tokens_a = []  # 第一段 Token 列表
                for j in range(a_end):
                    tokens_a.extend(document[i][j])  # 填充第一段数据

                tokens_b = []  # 第二段 Token 列表
                for j in range(a_end, len(document[i])):
                    tokens_b.extend(document[i][j])  # 填充第二段数据

                # 截断处理，确保两段加起来不超过最大长度
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                # --- 随机交换逻辑 ---
                if random.random() < 0.5:
                    # 情况 A：保持原序 (A1 -> A2)
                    label = 0  # Label 0: 同源，顺序正确
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))  # 添加 [CLS]
                    src.extend(tokens_a)  # 添加第一段
                    src.append(self.vocab.get(SEP_TOKEN))  # 添加 [SEP]
                    seg_pos = [len(src)]  # 记录第一段结束位置（用于 Segment Embedding）
                    src.extend(tokens_b)  # 添加第二段
                    src.append(self.vocab.get(SEP_TOKEN))  # 添加结尾 [SEP]
                    seg_pos.append(len(src))  # 记录第二段结束位置
                else:
                    # 情况 B：交换顺序 (A2 -> A1)
                    label = 1  # Label 1: 同源，顺序颠倒
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_b)  # 先放第二段
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_a)  # 后放第一段
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))

            # ==============================================================================
            # 分支 2：随机跳转/不同 Flow 任务 (主要对应 SODF Label 2)
            # 触发条件：随机数在 0.4 到 0.6 之间 (20% 概率)
            # ==============================================================================
            elif rnd < 0.6:
                tokens_a = []
                for j in range(len(document[i])):
                    tokens_a.extend(document[i][j])  # 当前 Burst 作为 tokens_a

                # 计算 tokens_b 允许的最大长度
                next_burst_max_length = target_seq_length - len(tokens_a)

                # --- 随机采样另一个 Burst ---
                for _ in range(20):  # 尝试 20 次寻找另一个 Flow
                    random_document_index = random.randint(0, len(all_documents) - 1)
                    if random_document_index != document_index:  # 尽量找不同的 Flow
                        break

                random_document = all_documents[random_document_index]  # 获取随机选中的 Flow
                random_start = random.randint(0, len(random_document) - 1)  # 在该 Flow 中随机选一个 Burst
                burst_ind_end = random_start + 1

                tokens_b = []
                for burst_ind in range(random_start, burst_ind_end):
                    for j in range(len(random_document[burst_ind])):
                        tokens_b.extend(random_document[burst_ind][j])  # 填充随机选中的 Burst 数据
                        if len(tokens_b) >= next_burst_max_length:
                            break

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)  # 截断

                # --- 确定标签 ---
                if random_document_index == document_index:
                    # 如果运气不好随机到了同一个 Flow (极低概率)，则退化为同 Flow 任务
                    if random_start >= i:
                        label = 3  # 后序 Burst
                    else:
                        label = 4  # 前序 Burst
                else:
                    label = 2  # Label 2: 来自完全不同的 Flow (这是本分支的主要目的)

                # 构建输入序列
                src = []
                src.append(self.vocab.get(CLS_TOKEN))
                src.extend(tokens_a)
                src.append(self.vocab.get(SEP_TOKEN))
                seg_pos = [len(src)]
                src.extend(tokens_b)
                src.append(self.vocab.get(SEP_TOKEN))
                seg_pos.append(len(src))

            # ==============================================================================
            # 分支 3：连续 Bursts 任务 (对应 SODF Label 3 & 4)
            # 触发条件：剩余情况 (40% 概率)
            # ==============================================================================
            else:
                tokens_a = []
                for j in range(len(document[i])):
                    tokens_a.extend(document[i][j])  # 当前 Burst 作为 tokens_a

                i += 1  # 索引 +1，获取下一个 Burst

                tokens_b = []
                for j in range(len(document[i])):
                    tokens_b.extend(document[i][j])  # 下一个 Burst 作为 tokens_b

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)  # 截断

                # --- 随机交换逻辑 ---
                if random.random() < 0.5:
                    # 情况 A：保持原序 (Burst 1 -> Burst 2)
                    label = 3  # Label 3: 同 Flow，连续，顺序正确
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
                else:
                    # 情况 B：交换顺序 (Burst 2 -> Burst 1)
                    label = 4  # Label 4: 同 Flow，连续，顺序颠倒
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))

            # ------------------------------------------------------------------
            # 公共后处理：Padding (填充)
            # ------------------------------------------------------------------
            while len(src) != self.seq_length:
                src.append(PAD_ID)  # 用 [PAD] 填充至固定长度

            # ------------------------------------------------------------------
            # 公共后处理：MLM Masking (掩码)
            # ------------------------------------------------------------------
            if not self.dynamic_masking:
                # 执行 Mask 操作，随机将部分 Token 替换为 [MASK]
                # 返回 src (Mask 后) 和 tgt_mlm (被 Mask 的真实值)
                src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking,
                                        self.span_geo_prob, self.span_max_length)
                instance = (src, tgt_mlm, label, seg_pos)
            else:
                instance = (src, label, seg_pos)

            # 将协议类型 (proto) 加到样本末尾，供 MoE 或其他机制使用
            instance += (flow_proto[document_index],)

            instances.append(instance)  # 加入样本集
            i += 1  # 继续处理下一个 Burst (注意分支 3 里面已经加过一次了，这里会再次加，跳过已处理的 next burst)

        return instances


# 继承自 DataLoader，用于加载 BertDataset 生成的数据。返回批量的输入序列、掩码标签、NSP 标签和段落标记。
class BertDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_mlm = []
            is_next = []
            seg = []

            masked_words_num = 0

            for ins in instances:
                if len(ins) == 4:
                    src.append(ins[0])
                    masked_words_num += len(ins[1])
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[2])
                    seg.append([1] * ins[3][0] + [2] * (ins[3][1] - ins[3][0]) + [PAD_ID] * (len(ins[0]) - ins[3][1]))
                else:
                    src_single, tgt_mlm_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking,
                                                          self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_mlm_single)
                    src.append(src_single)
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[1])
                    seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [PAD_ID] * (len(ins[0]) - ins[2][1]))

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(is_next), \
                torch.LongTensor(seg)


# 继承自 DataLoader，用于加载 BertFlowDataset 生成的数据。除了常规字段外，还返回协议类型标签。
class BertFlowDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_mlm = []
            is_next = []
            seg = []
            proto = []

            masked_words_num = 0

            for ins in instances:
                if len(ins) == 5:
                    src.append(ins[0])
                    masked_words_num += len(ins[1])
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[2])
                    seg.append([1] * ins[3][0] + [2] * (ins[3][1] - ins[3][0]) + [PAD_ID] * (len(ins[0]) - ins[3][1]))
                    proto.append(ins[4])
                else:
                    src_single, tgt_mlm_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking,
                                                          self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_mlm_single)
                    src.append(src_single)
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[1])
                    seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [PAD_ID] * (len(ins[0]) - ins[2][1]))
                    proto.append(ins[4])
            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(is_next), \
                torch.LongTensor(seg), \
                torch.LongTensor(proto)


# 继承自 Dataset，仅用于 MLM 任务。支持是否将整个句子作为一个文档处理，生成掩码后的序列和标签。
class MlmDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(MlmDataset, self).__init__(args, vocab, tokenizer)
        self.full_sentences = args.full_sentences

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)

        # --- 修改开始 ---
        # 使用绝对路径
        abs_dataset_path = os.path.abspath(self.dataset_path)
        temp_dir = os.path.join(os.path.dirname(abs_dataset_path), "temp_datasets")

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

        # 打印一下路径，方便调试
        temp_file = os.path.join(temp_dir, "dataset-tmp-" + str(proc_id) + ".pt")
        print(f"Worker {proc_id} writing to: {temp_file}")

        dataset_writer = open(temp_file, "wb")
        # --- 修改结束 ---

        docs_buffer = []
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    pos += 1

                    document = [self.vocab.get(CLS_TOKEN)] + self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(line)) + [self.vocab.get(SEP_TOKEN)]

                    if self.full_sentences:
                        if len(document) > 0:
                            docs_buffer.append(document)
                        if len(docs_buffer) == self.docs_buffer_size:
                            # Build instances from documents.
                            all_documents = self.concatenate_docs(docs_buffer)
                            instances = self.build_instances(all_documents)
                            # Save instances.
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                            # Clear buffer.
                            docs_buffer = []
                        if pos >= end:
                            if len(docs_buffer) > 0:
                                all_documents = self.concatenate_docs(docs_buffer)
                                instances = self.build_instances(all_documents)
                                # Save instances.
                                for instance in instances:
                                    pickle.dump(instance, dataset_writer)
                            break
                    else:
                        if len(document) > 0:
                            instances = self.build_instances(document)
                            # Save instances.
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)

                    if pos >= end:
                        break

        dataset_writer.close()

    def concatenate_docs(self, docs_buffer):
        all_documents = []
        for i in range(len(docs_buffer)):
            all_documents += docs_buffer[i]
        return all_documents

    def build_instances(self, all_documents):
        instances = []
        instances_num = len(all_documents) // self.seq_length
        for i in range(instances_num):
            src = all_documents[i * self.seq_length: (i + 1) * self.seq_length]
            seg_pos = [len(src)]

            if not self.dynamic_masking:
                src, tgt = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob,
                                    self.span_max_length)
                instance = (src, tgt, seg_pos)
            else:
                instance = (src, seg_pos)

            instances.append(instance)

        src = all_documents[instances_num * self.seq_length:]
        seg_pos = [len(src)]

        while len(src) != self.seq_length:
            src.append(PAD_ID)

        if not self.dynamic_masking:
            src, tgt = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob,
                                self.span_max_length)
            instance = (src, tgt, seg_pos)
        else:
            instance = (src, seg_pos)

        instances.append(instance)
        return instances


# 继承自 DataLoader，用于加载 MlmDataset 生成的数据。返回批量的输入序列、掩码标签和段落标记。
class MlmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            masked_words_num = 0

            for ins in instances:
                if len(ins) == 3:
                    src.append(ins[0])
                    masked_words_num += len(ins[1])
                    tgt.append([0] * len(ins[0]))
                    for mask in ins[1]:
                        tgt[-1][mask[0]] = mask[1]
                    seg.append([1] * ins[2][0] + [PAD_ID] * (len(ins[0]) - ins[2][0]))
                else:
                    src_single, tgt_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking,
                                                      self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_single)
                    src.append(src_single)
                    tgt.append([0] * len(ins[0]))
                    for mask in tgt_single:
                        tgt[-1][mask[0]] = mask[1]
                    seg.append([1] * ins[1][0] + [PAD_ID] * (len(ins[0]) - ins[1][0]))

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


# 继承自 Dataset，用于构建 ALBERT 模型的数据集，支持 MLM 和 SOP（句子顺序预测）任务。生成句子对并随机打乱顺序以构建正负样本。
class AlbertDataset(Dataset):
    """
    Construct dataset for MLM and SOP tasks from the given corpus.
    Each document consists of multiple sentences,
    and each sentence occupies a single line.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(AlbertDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        document = []
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.path.dirname(self.dataset_path), "temp_datasets")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        dataset_writer = open(os.path.join(temp_dir, "dataset-tmp-" + str(proc_id) + ".pt"), "wb")
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    pos += 1
                    if not line.strip():
                        if len(document) >= 1:
                            instances = self.build_instances(document)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        document = []
                    sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                    if len(sentence) > 0:
                        document.append(sentence)
                    if pos >= end - 1:
                        if len(document) >= 1:
                            instances = self.build_instances(document)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        break
        dataset_writer.close()

    def build_instances(self, document):
        instances = []
        instances.extend(self.create_ins_from_doc(document))
        return instances

    def create_ins_from_doc(self, document):
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    is_wrong_order = 0
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                    if random.random() < 0.5:
                        is_wrong_order = 1
                        tmp = tokens_a
                        tokens_a = tokens_b
                        tokens_b = tmp

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))

                    while len(src) != self.seq_length:
                        src.append(PAD_ID)

                    if not self.dynamic_masking:
                        src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking,
                                                self.span_geo_prob, self.span_max_length)
                        instance = (src, tgt_mlm, is_wrong_order, seg_pos)
                    else:
                        instance = (src, is_wrong_order, seg_pos)

                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances


# 继承自 BertDataLoader，复用 BERT 的数据加载逻辑，用于加载 AlbertDataset 数据。
class AlbertDataLoader(BertDataLoader):
    '''
    AlbertDataLoader can reuse the code of BertDataLoader.
    '''
    pass


# 继承自 Dataset，用于语言模型（LM）任务。将文本转换为 token 序列，并分割为固定长度的片段，用于下一个词预测。
class LmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.path.dirname(self.dataset_path), "temp_datasets")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        dataset_writer = open(os.path.join(temp_dir, "dataset-tmp-" + str(proc_id) + ".pt"), "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                document = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                document = [self.vocab.get(CLS_TOKEN)] + document + [self.vocab.get(SEP_TOKEN)]

                instances_num = len(document) // (self.seq_length + 1)
                for i in range(instances_num):
                    src = document[i * (self.seq_length + 1): (i + 1) * (self.seq_length + 1)]
                    seg_pos = self.seq_length
                    pickle.dump((src, seg_pos), dataset_writer)

                src = document[instances_num * (self.seq_length + 1):]
                if len(src) > 0:
                    seg_pos = len(src)
                    while len(src) != self.seq_length + 1:
                        src.append(PAD_ID)
                    pickle.dump((src, seg_pos), dataset_writer)

                if pos >= end:
                    break

        dataset_writer.close()


# 继承自 DataLoader，用于加载 LmDataset 数据。返回输入序列、目标序列（右移一位）和段落标记。
class LmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0][:-1])
                tgt.append(ins[0][1:])
                if ins[1] == len(ins[0]):
                    seg.append([1] * (ins[1] - 1))
                else:
                    seg.append([1] * ins[1] + [PAD_ID] * (len(ins[0]) - 1 - ins[1]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


# 继承自 Dataset，用于双向语言模型（BiLM）任务。生成同时包含前向和后向语言模型目标的数据。
class BilmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.path.dirname(self.dataset_path), "temp_datasets")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        dataset_writer = open(os.path.join(temp_dir, "dataset-tmp-" + str(proc_id) + ".pt"), "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                document = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))

                instances_num = len(document) // self.seq_length
                for i in range(instances_num):
                    src = document[i * self.seq_length: (i + 1) * self.seq_length]
                    tgt_forward = src[1:] + [self.vocab.get(SEP_TOKEN)]
                    tgt_backward = [self.vocab.get(CLS_TOKEN)] + src[:-1]
                    seg = [1] * len(src)
                    pickle.dump((src, tgt_forward, tgt_backward, seg), dataset_writer)

                src = document[instances_num * self.seq_length:]
                if len(src) < 1:
                    continue
                tgt_forward = src[1:] + [self.vocab.get(SEP_TOKEN)]
                tgt_backward = [self.vocab.get(CLS_TOKEN)] + src[:-1]
                seg = [1] * len(src)
                while len(src) != self.seq_length:
                    src.append(PAD_ID)
                    tgt_forward.append(PAD_ID)
                    tgt_backward.append(PAD_ID)
                    seg.append(PAD_ID)
                pickle.dump((src, tgt_forward, tgt_backward, seg), dataset_writer)

                if pos >= end - 1:
                    break

        dataset_writer.close()


# 继承自 DataLoader，用于加载 BilmDataset 数据。返回输入序列、前向目标、后向目标和段落标记。
class BilmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_forward = []
            tgt_backward = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt_forward.append(ins[1])
                tgt_backward.append(ins[2])
                seg.append(ins[3])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_forward), \
                torch.LongTensor(tgt_backward), \
                torch.LongTensor(seg)


# 继承自 Dataset，用于序列到序列（Seq2Seq）任务。读取源文本和目标文本，分别转换为 token 序列，并填充到固定长度。
class Seq2seqDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(Seq2seqDataset, self).__init__(args, vocab, tokenizer)
        self.tgt_seq_length = args.tgt_seq_length
        self.src_vocab, self.src_tokenizer = vocab, tokenizer
        self.tgt_tokenizer = args.tgt_tokenizer
        self.tgt_vocab = self.tgt_tokenizer.vocab

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.path.dirname(self.dataset_path), "temp_datasets")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        dataset_writer = open(os.path.join(temp_dir, "dataset-tmp-" + str(proc_id) + ".pt"), "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                if len(line.strip().split("\t")) != 2:
                    if pos >= end:
                        break
                    continue
                document_src, document_tgt = line.strip().split("\t")
                src = self.src_tokenizer.convert_tokens_to_ids(self.src_tokenizer.tokenize(document_src))
                tgt = self.tgt_tokenizer.convert_tokens_to_ids(self.tgt_tokenizer.tokenize(document_tgt))

                src = [self.src_vocab.get(CLS_TOKEN)] + src + [self.src_vocab.get(SEP_TOKEN)]
                tgt = [self.tgt_vocab.get(CLS_TOKEN)] + tgt + [self.tgt_vocab.get(SEP_TOKEN)]
                seg = [1] * len(src)

                src, tgt, seg = src[:self.seq_length], tgt[:self.tgt_seq_length + 1], seg[:self.seq_length]
                while len(src) != self.seq_length:
                    src.append(PAD_ID)
                    seg.append(PAD_ID)
                while len(tgt) != self.tgt_seq_length + 1:
                    tgt.append(PAD_ID)
                pickle.dump((src, tgt, seg), dataset_writer)

                if pos >= end:
                    break

            dataset_writer.close()


# 继承自 DataLoader，用于加载 Seq2seqDataset 数据。返回源序列、目标输入（去尾）、目标输出（去头）和段落标记。
class Seq2seqDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_in = []
            tgt_out = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt_in.append(ins[1][:-1])
                tgt_out.append(ins[1][1:])
                seg.append(ins[2])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg)


# 继承自 MlmDataset，复用 MLM 数据构建逻辑，用于 T5 模型的预训练任务。
class T5Dataset(MlmDataset):
    '''
    T5 can reuse the code of MlmDataset.
    '''
    pass


# 继承自 DataLoader，用于加载 T5Dataset 数据。将掩码后的序列转换为 T5 格式的输入和目标序列，使用哨兵 token 表示被掩码的部分。
class T5DataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_in = []
            tgt_out = []
            seg = []

            tgt_seq_length = 0

            for _, ins in enumerate(instances):
                if len(ins) == 3:
                    src_single = ins[0]
                    tgt_single = ins[1]
                    seg.append([1] * ins[2][0] + [PAD_ID] * (len(ins[0]) - ins[2][0]))
                else:
                    src_single, tgt_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking,
                                                      self.span_masking, self.span_geo_prob, self.span_max_length)
                    seg.append([1] * ins[1][0] + [PAD_ID] * (len(ins[0]) - ins[1][0]))

                MASK_ID = self.vocab.get(MASK_TOKEN)
                SENTINEL_ID = self.vocab.get(SENTINEL_TOKEN)

                for src_index, _ in tgt_single:
                    if src_single[src_index] != MASK_ID:
                        src_single[src_index] = MASK_ID

                tgt_in_single = [self.vocab.get(CLS_TOKEN)]
                mask_index = 0
                src_with_sentinel = []
                for token_id in src_single:
                    if token_id == MASK_ID:
                        if len(src_with_sentinel) > 0 and src_with_sentinel[-1] == (SENTINEL_ID - 1):
                            pass
                        else:
                            src_with_sentinel.append(SENTINEL_ID)
                            tgt_in_single.append(SENTINEL_ID)
                            SENTINEL_ID += 1
                        tgt_in_single.append(tgt_single[mask_index][1])
                        mask_index += 1
                    else:
                        src_with_sentinel.append(token_id)
                tgt_in_single.append(SENTINEL_ID)
                tgt_in_single.append(self.vocab.get(SEP_TOKEN))

                while len(src_with_sentinel) < len(src_single):
                    src_with_sentinel.append(PAD_ID)

                if len(tgt_in_single) > tgt_seq_length:
                    tgt_seq_length = len(tgt_in_single)

                src.append(src_with_sentinel)
                tgt_in.append(tgt_in_single)
                tgt_out.append(tgt_in[-1][1:] + [PAD_ID])

            for i in range(len(tgt_in)):
                while len(tgt_in[i]) != tgt_seq_length:
                    tgt_in[i].append(PAD_ID)
                    tgt_out[i].append(PAD_ID)

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg)


# 继承自 Dataset，用于文本分类任务。读取标签和文本，构建输入序列和段落标记，支持单句和句对输入。
class ClsDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.path.dirname(self.dataset_path), "temp_datasets")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        f_write = open(os.path.join(temp_dir, "dataset-tmp-" + str(proc_id) + ".pt"), "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                line = f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                line = line.strip().split('\t')
                if len(line) == 2:
                    label = int(line[0])
                    text = " ".join(line[1:])
                    src = [self.vocab.get(t) for t in self.tokenizer.tokenize(text)]
                    src = [self.vocab.get(CLS_TOKEN)] + src
                    tgt = label
                    seg = [1] * len(src)
                    if len(src) >= self.seq_length:
                        src = src[:self.seq_length]
                        seg = seg[:self.seq_length]
                    else:
                        while len(src) != self.seq_length:
                            src.append(PAD_ID)
                            seg.append(PAD_ID)
                    pickle.dump((src, tgt, seg), f_write)
                elif len(line) == 3:  # For sentence pair input.
                    label = int(line[0])
                    text_a, text_b = line[1], line[2]

                    src_a = [self.vocab.get(t) for t in self.tokenizer.tokenize(text_a)]
                    src_a = [self.vocab.get(CLS_TOKEN)] + src_a + [self.vocab.get(SEP_TOKEN)]
                    src_b = [self.vocab.get(t) for t in self.tokenizer.tokenize(text_b)]
                    src_b = src_b + [self.vocab.get(SEP_TOKEN)]

                    src = src_a + src_b
                    seg = [1] * len(src_a) + [2] * len(src_b)

                    if len(src) >= self.seq_length:
                        src = src[:self.seq_length]
                        seg = seg[:self.seq_length]
                    else:
                        while len(src) != self.seq_length:
                            src.append(PAD_ID)
                            seg.append(PAD_ID)
                    pickle.dump((src, tgt, seg), f_write)
                else:
                    pass

                if pos >= end - 1:
                    break

        f_write.close()


# 继承自 DataLoader，用于加载 ClsDataset 数据。返回输入序列、分类标签和段落标记。
class ClsDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append(ins[2])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


# 继承自 Dataset，用于前缀语言模型任务。将输入分为前缀和后缀，构建自回归训练数据。
class PrefixlmDataset(Dataset):

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.path.dirname(self.dataset_path), "temp_datasets")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        dataset_writer = open(os.path.join(temp_dir, "dataset-tmp-" + str(proc_id) + ".pt"), "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                if len(line.strip().split("\t")) != 2:
                    if pos >= end:
                        break
                    continue
                document_src, document_tgt = line.strip().split("\t")
                src = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(document_src))
                tgt = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(document_tgt))
                src = [self.vocab.get(CLS_TOKEN)] + src + [self.vocab.get(SEP_TOKEN)]
                tgt = tgt + [self.vocab.get(SEP_TOKEN)]
                seg_pos = [len(src)]

                if seg_pos[0] >= self.seq_length:
                    continue

                src = src + tgt
                tgt = [0] * seg_pos[0] + tgt[1:] + [PAD_ID]
                seg_pos.append(len(src))
                src, tgt = src[:self.seq_length], tgt[:self.seq_length]
                while len(src) != self.seq_length:
                    src.append(PAD_ID)
                    tgt.append(PAD_ID)
                if seg_pos[1] > self.seq_length:
                    seg_pos[1] = self.seq_length

                pickle.dump((src, tgt, seg_pos), dataset_writer)

                if pos >= end:
                    break

            dataset_writer.close()


# 继承自 DataLoader，用于加载 PrefixlmDataset 数据。返回输入序列、目标序列和段落标记。
class PrefixlmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [PAD_ID] * (len(ins[0]) - ins[2][1]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)
