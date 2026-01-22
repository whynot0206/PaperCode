"""
This script provides an exmaple to wrap UER-py for classification.
"""
import os  # 导入操作系统接口模块
import sys  # 导入系统相关参数和函数模块

sys.path.append(os.getcwd())  # 将当前工作目录添加到系统路径
import random  # 导入随机数生成模块
import argparse  # 导入命令行参数解析模块
import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch神经网络模块
from uer.layers import *  # 从UER导入所有层
from uer.encoders import *  # 从UER导入所有编码器
from uer.utils.vocab import Vocab  # 从UER导入词汇表类
from uer.utils.constants import *  # 从UER导入所有常量
from uer.utils import *  # 从UER导入所有工具函数
from uer.utils.optimizers import *  # 从UER导入所有优化器
from uer.utils.config import load_hyperparam  # 从UER导入超参数加载函数
from uer.utils.seed import set_seed  # 从UER导入随机种子设置函数
from uer.model_saver import save_model  # 从UER导入模型保存函数
from uer.opts import finetune_opts  # 从UER导入微调选项
from uer.targets import MlmTarget  # 从UER导入MLM目标
import tqdm  # 导入进度条显示模块
import numpy as np  # 导入数值计算库
from sklearn.metrics import f1_score, precision_score, recall_score  # 从sklearn导入评估指标


class Classifier(nn.Module):  # 定义分类器类，继承自nn.Module
    def __init__(self, args):  # 初始化函数
        super(Classifier, self).__init__()  # 调用父类初始化函数
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))  # 根据参数创建嵌入层
        self.encoder = str2encoder[args.encoder](args)  # 根据参数创建编码器
        self.target = MlmTarget(args, len(args.tokenizer.vocab))  # 创建MLM目标

    def forward(self, src, tgt, seg, soft_tgt=None):  # 前向传播函数
        """
        Args:
            src: [batch_size x seq_length]  # 输入序列
            tgt: [batch_size x seq_length]  # 目标序列（用于MLM）
            seg: [batch_size x seq_length]  # 分段标识
        """
        # Embedding.
        emb = self.embedding(src, seg)  # 通过嵌入层获取嵌入表示
        # Encoder.
        output = self.encoder(emb, seg)  # 通过编码器获取编码输出
        loss_mlm, output_mlm, tgt_mlm = self.target.mlm2(output, tgt)  # 计算MLM损失、输出和目标
        return loss_mlm, output_mlm, tgt_mlm  # 返回MLM损失、输出和目标


def load_or_initialize_parameters(args, model):  # 加载或初始化模型参数的函数
    if args.pretrained_model_path is not None:  # 如果提供了预训练模型路径
        print("Initialize with pretrained model.")  # 打印信息
        model.load_state_dict(torch.load(args.pretrained_model_path,
                                         map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0', 'cuda:3': 'cuda:0'}),
                              strict=False)  # 加载预训练模型参数
    else:  # 如果没有提供预训练模型
        print("Initialize with normal distribution.")  # 打印信息
        for n, p in list(model.named_parameters()):  # 遍历所有参数
            if "gamma" not in n and "beta" not in n:  # 如果不是gamma或beta参数（LayerNorm参数）
                p.data.normal_(0, 0.02)  # 使用正态分布初始化参数


def build_optimizer(args, model):  # 构建优化器的函数
    param_optimizer = list(model.named_parameters())  # 获取所有参数名称和参数
    no_decay = ['bias', 'gamma', 'beta']  # 定义不需要权重衰减的参数名称
    optimizer_grouped_parameters = [  # 分组优化器参数
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        # 需要权重衰减的参数
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        # 不需要权重衰减的参数
    ]
    if args.optimizer in ["adamw"]:  # 如果使用AdamW优化器
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  correct_bias=False)  # 创建AdamW优化器
    else:  # 如果使用其他优化器
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)  # 创建其他优化器
    if args.scheduler in ["constant"]:  # 如果使用恒定学习率调度器
        scheduler = str2scheduler[args.scheduler](optimizer)  # 创建恒定学习率调度器
    elif args.scheduler in ["constant_with_warmup"]:  # 如果使用带热身的恒定学习率调度器
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)  # 创建带热身的恒定学习率调度器
    else:  # 如果使用其他调度器
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup,
                                                  args.train_steps)  # 创建其他调度器
    return optimizer, scheduler  # 返回优化器和调度器


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):  # 批量数据加载器函数
    instances_num = src.size()[0]  # 获取实例数量
    for i in range(instances_num // batch_size):  # 遍历完整批次
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]  # 获取源数据批次
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]  # 获取目标数据批次
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]  # 获取分段标识批次
        if soft_tgt is not None:  # 如果有软目标
            soft_tgt_batch = soft_tgt[i * batch_size: (i + 1) * batch_size, :]  # 获取软目标批次
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch  # 返回批次数据（含软目标）
        else:  # 如果没有软目标
            yield src_batch, tgt_batch, seg_batch, None  # 返回批次数据（不含软目标）
    if instances_num > instances_num // batch_size * batch_size:  # 如果有剩余数据
        src_batch = src[instances_num // batch_size * batch_size:, :]  # 获取剩余源数据
        tgt_batch = tgt[instances_num // batch_size * batch_size:]  # 获取剩余目标数据
        seg_batch = seg[instances_num // batch_size * batch_size:, :]  # 获取剩余分段标识
        if soft_tgt is not None:  # 如果有软目标
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size:, :]  # 获取剩余软目标
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch  # 返回剩余批次数据（含软目标）
        else:  # 如果没有软目标
            yield src_batch, tgt_batch, seg_batch, None  # 返回剩余批次数据（不含软目标）


def read_dataset(args, path):  # read data with SEP  # 读取数据集的函数（带SEP分隔符）
    dataset, columns = [], {}  # 初始化数据集和列字典
    with open(path, mode="r", encoding="utf-8") as f:  # 打开文件
        for line_id, line in enumerate(f):  # 遍历文件行
            if line_id == 0:  # 如果是第一行（表头）
                for i, column_name in enumerate(line.strip().split("\t")):  # 遍历列名
                    columns[column_name] = i  # 记录列索引
                continue  # 继续下一行
            line = line[:-1].split("\t")  # 分割行数据（去除换行符）

            if "text_b" in columns:  # 如果有text_b列
                print("error, only one sentence")  # 打印错误信息（只支持单句）

            text_a = line[columns["text_a"]]  # 获取文本a
            text_list = text_a.split("[SEP]")[1:]  # 分割文本列表（去除第一个空元素）
            if text_list[0].split(" ")[1:-1][9][:2] != "06":  # only handle TCP  # 如果不是TCP协议（只处理TCP）
                continue  # 跳过当前行
            field_cover = {"IPID": [3, 4, 5], "srcip": [11, 12, 13, 14, 15], "dstip": [15, 16, 17, 18, 19],
                           "srcport": [19, 20, 21], "dstport": [21, 22, 23],  # 定义字段覆盖范围
                           "seq": [23, 24, 25, 26, 27], "ack": [27, 28, 29, 30, 31], "hdrlen": [31, 32],
                           "tcpflags": [32, 33]}  # 继续定义字段覆盖范围
            mask_fields = ["IPID", "srcip", "dstip", "srcport", "dstport", "seq", "ack", "hdrlen",
                           "tcpflags"]  # 定义需要掩码的字段
            dir_fields = ["srcip", "dstip", "srcport", "dstport"]  # 定义方向字段
            mask_index = []  # 初始化掩码索引列表
            for key in mask_fields:  # 遍历掩码字段
                mask_index.extend(field_cover[key])  # 扩展掩码索引列表
            for key in random.sample(dir_fields, 1):  # 随机选择一个方向字段
                for j in field_cover[key]:  # 遍历该方向字段的覆盖范围
                    mask_index.remove(j)  # 从掩码索引中移除（不掩码方向字段）
            datagramss = []  # 初始化数据报文列表
            mask_index_in_datagramss = []  # 初始化数据报文中的掩码索引列表
            for i in range(len(text_list)):  # 遍历文本列表
                pac = text_list[i]  # 获取数据包文本
                datagrams = pac.split(" ")[1:-1]  # 分割数据报文（去除首尾空格）
                for j in range(len(datagrams)):  # 遍历数据报文元素
                    if i == len(text_list) - 1:  # 如果是最后一个数据包
                        if j in mask_index:  # 如果元素索引在掩码索引中
                            mask_index_in_datagramss.append(len(datagramss))  # 添加到数据报文掩码索引列表
                datagramss.append(datagrams)  # 添加到数据报文列表
            # print(datagramss)  # 注释掉的调试信息
            # print(mask_index_in_datagramss)  # 注释掉的调试信息
            # for i in mask_index_in_datagramss:  # 注释掉的调试信息
            #     print(datagramss[i])  # 注释掉的调试信息
            newtext_a = ''  # 初始化新文本
            for i in range(len(datagramss)):  # 遍历数据报文列表
                if newtext_a != '':  # 如果不是第一个数据包
                    newtext_a += ' '  # 添加空格
                newtext_a += datagramss[i]  # 添加数据报文
            text_a_tokens = args.tokenizer.tokenize(newtext_a)  # 对新文本进行分词

            mask_index_in_tokens = []  # 初始化分词中的掩码索引列表
            word_ind = 0  # 初始化单词索引
            token_ind = 0  # 初始化分词索引
            while word_ind < len(datagramss):  # 遍历数据报文
                # if datagramss[word_ind] == text_a_tokens[token_ind]:  # 注释掉的代码：如果数据报文元素等于分词
                #     word_ind += 1  # 注释掉的代码：单词索引加1
                #     token_ind += 1  # 注释掉的代码：分词索引加1
                # else:  # 注释掉的代码：否则
                temp = text_a_tokens[token_ind].replace('#', '')  # 获取当前分词（去除#号）
                if word_ind in mask_index_in_datagramss:  # 如果单词索引在数据报文掩码索引中
                    mask_index_in_tokens.append(token_ind)  # 添加到分词掩码索引列表
                while datagramss[word_ind] != temp:  # 当数据报文元素不等于当前分词
                    token_ind += 1  # 分词索引加1
                    temp += text_a_tokens[token_ind].replace('#', '')  # 拼接下一个分词
                    if word_ind in mask_index_in_datagramss:  # 如果单词索引在数据报文掩码索引中
                        mask_index_in_tokens.append(token_ind)  # 添加到分词掩码索引列表
                word_ind += 1  # 单词索引加1
                token_ind += 1  # 分词索引加1

            # for i in mask_index_in_tokens:  # 注释掉的调试信息
            #     print(text_a_tokens[i])  # 注释掉的调试信息
            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + text_a_tokens)  # 将分词转换为ID序列（添加[CLS]）
            # print(src)  # 注释掉的调试信息
            tgt = [0] * len(src)  # 初始化目标序列（全0）
            for i in mask_index_in_tokens:  # 遍历分词掩码索引
                tgt[i + 1] = src[i + 1]  # consider CLS  # 设置目标序列（考虑[CLS]位置）
                src[i + 1] = args.tokenizer.vocab.get(MASK_TOKEN)  # 将源序列中的对应位置替换为[MASK]
            seg = [1] * len(src)  # 创建分段标识（全为1）
            # print(src)  # 注释掉的调试信息
            # print(tgt)  # 注释掉的调试信息

            if len(src) > args.seq_length:  # 如果序列长度超过最大长度
                src = src[: args.seq_length]  # 截断序列
                seg = seg[: args.seq_length]  # 截断分段标识
                tgt = tgt[: args.seq_length]  # 截断目标序列
            while len(src) < args.seq_length:  # 如果序列长度小于最大长度
                src.append(0)  # 填充0
                seg.append(0)  # 填充0
                tgt.append(0)  # 填充0

            dataset.append((src, tgt, seg))  # 添加数据到数据集

    return dataset  # 返回数据集


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):  # 训练模型的函数
    model.zero_grad()  # 清除梯度

    src_batch = src_batch.to(args.device)  # 将源数据移动到设备
    tgt_batch = tgt_batch.to(args.device)  # 将目标数据移动到设备
    seg_batch = seg_batch.to(args.device)  # 将分段标识移动到设备
    if soft_tgt_batch is not None:  # 如果有软目标批次
        soft_tgt_batch = soft_tgt_batch.to(args.device)  # 将软目标移动到设备

    loss, _, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)  # 前向传播计算损失
    if torch.cuda.device_count() > 1:  # 如果使用多个GPU
        loss = torch.mean(loss)  # 对损失求均值（多GPU情况）

    if args.fp16:  # 如果使用混合精度训练
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:  # 缩放损失
            scaled_loss.backward()  # 反向传播
    else:  # 如果不使用混合精度训练
        loss.backward()  # 反向传播

    optimizer.step()  # 更新参数
    scheduler.step()  # 更新学习率

    return loss  # 返回损失值


def evaluate(args, dataset, print_confusion_matrix=False):  # 评估模型的函数
    src = torch.LongTensor([sample[0] for sample in dataset])  # 创建源数据张量
    tgt = torch.LongTensor([sample[1] for sample in dataset])  # 创建目标数据张量
    seg = torch.LongTensor([sample[2] for sample in dataset])  # 创建分段标识张量

    batch_size = args.batch_size  # 获取批次大小

    correct = 0  # 初始化正确预测计数
    # Confusion matrix.
    if print_confusion_matrix:  # 如果需要打印混淆矩阵
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)  # 初始化混淆矩阵
    y_true, y_pred = [], []  # 初始化真实标签和预测标签列表
    args.model.eval()  # 设置模型为评估模式

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):  # 遍历批次数据
        src_batch = src_batch.to(args.device)  # 将源数据批次移动到设备
        tgt_batch = tgt_batch.to(args.device)  # 将目标数据批次移动到设备
        seg_batch = seg_batch.to(args.device)  # 将分段标识批次移动到设备
        with torch.no_grad():  # 禁用梯度计算
            _, output_mlm, tgt_mlm = args.model(src_batch, tgt_batch, seg_batch)  # 前向传播获取MLM输出和目标
        pred = output_mlm.argmax(dim=-1)  # 获取预测结果（取最大概率的token）
        gold = tgt_mlm  # 获取真实目标
        # for j in range(pred.size()[0]):  # 注释掉的代码：遍历批次中的每个样本
        #     if print_confusion_matrix:  # 注释掉的代码：如果需要打印混淆矩阵
        #         confusion[pred[j], gold[j]] += 1  # 注释掉的代码：更新混淆矩阵
        y_true.extend(gold.cpu())  # 添加真实目标到列表
        y_pred.extend(pred.cpu())  # 添加预测目标到列表
        correct += torch.sum(pred == gold).item()  # 更新正确预测计数

    if print_confusion_matrix:  # 如果需要打印混淆矩阵
        print("Confusion matrix:")  # 打印混淆矩阵标题
        print(confusion)  # 打印混淆矩阵
        cf_array = confusion.numpy()  # 将混淆矩阵转换为numpy数组
        # with open("./results/confusion_matrix",'w') as f:  # 注释掉的代码：保存混淆矩阵到文件
        #     for cf_a in cf_array:  # 注释掉的代码：遍历数组
        #         f.write(str(cf_a)+'\n')  # 注释掉的代码：写入文件
        print("Report precision, recall, and f1:")  # 打印评估指标标题
        eps = 1e-9  # 定义小值防止除零
        for i in range(confusion.size()[0]):  # 遍历每个类别
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)  # 计算精确率
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)  # 计算召回率
            if (p + r) == 0:  # 如果精确率和召回率都为0
                f1 = 0  # F1分数为0
            else:  # 如果精确率和召回率不都为0
                f1 = 2 * p * r / (p + r)  # 计算F1分数
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))  # 打印每个类别的评估指标

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(y_true), correct, len(y_true)))  # 打印准确率
    # print("Macro precision: {:.4f}, Micro precision: {:.4f}, Weighted precision: {:.4f}".format(  # 注释掉的代码：打印各种精确率
    #     precision_score(y_true,y_pred,average='macro'), precision_score(y_true,y_pred,average='micro'), precision_score(y_true,y_pred,average='weighted')))  # 注释掉的代码
    # print("Macro recall: {:.4f}, Micro recall: {:.4f}, Weighted recall: {:.4f}".format(  # 注释掉的代码：打印各种召回率
    #     recall_score(y_true,y_pred,average='macro'), recall_score(y_true,y_pred,average='micro'), recall_score(y_true,y_pred,average='weighted')))  # 注释掉的代码
    # print("Macro f1: {:.4f}, Micro f1: {:.4f}, Weighted f1: {:.4f}".format(  # 注释掉的代码：打印各种F1分数
    #     f1_score(y_true,y_pred,average='macro'), f1_score(y_true,y_pred,average='micro'), f1_score(y_true,y_pred,average='weighted')))  # 注释掉的代码

    return correct / len(y_true)  # 返回准确率


def main():  # 主函数
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 创建参数解析器

    finetune_opts(parser)  # 添加微调选项

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",  # 添加池化方式参数
                        help="Pooling type.")  # 帮助信息：池化类型

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",  # 添加分词器参数
                        help="Specify the tokenizer."  # 帮助信息：指定分词器
                             "Original Google BERT uses bert tokenizer on Chinese corpus."  # 原始Google BERT在中文语料上使用bert分词器
                             "Char tokenizer segments sentences into characters."  # 字符分词器将句子分割成字符
                             "Space tokenizer segments sentences into words according to space."  # 空格分词器根据空格将句子分割成单词
                        )

    parser.add_argument("--soft_targets", action='store_true',  # 添加软目标参数
                        help="Train model with logits.")  # 帮助信息：使用logits训练模型
    parser.add_argument("--soft_alpha", type=float, default=0.5,  # 添加软目标权重参数
                        help="Weight of the soft targets loss.")  # 帮助信息：软目标损失的权重

    # MOE Model Options
    parser.add_argument("--is_moe", action="store_true", help="adopt moe layer.")  # 添加是否使用MOE层参数
    parser.add_argument("--vocab_size", type=int, required=False, help="Number of vocab.")  # 添加词汇表大小参数
    parser.add_argument("--moebert_expert_dim", type=int, required=False, default=3072,
                        help="Dim of expert,default is ffn.")  # 添加MOE专家维度参数
    parser.add_argument("--moebert_expert_num", type=int, required=False, help="Number of expert.")  # 添加MOE专家数量参数
    parser.add_argument("--moebert_route_method",
                        choices=["gate-token", "gate-sentence", "hash-random", "hash-balance", "proto"],
                        default="hash-random",  # 添加MOE路由方法参数
                        help="moebert route method.")  # 帮助信息：MOE路由方法
    parser.add_argument("--moebert_route_hash_list", default=None, type=str,
                        help="Path of moebert hash list file.")  # 添加MOE哈希列表路径参数
    parser.add_argument("--moebert_load_balance", type=float, default=0.0, help="gate loss weight.")  # 添加MOE负载平衡参数

    args = parser.parse_args()  # 解析参数

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)  # 从配置文件加载超参数

    set_seed(args.seed)  # 设置随机种子

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)  # 构建分词器

    # text_a = "4500 0002 029d 9d5d 5df7 f740 4000 007f 7f06 06fd fd31 3196 96f2 f2a9 a964 6475 7512 12e8 e8c8 c812 123d 3d01 01bb bbb6 b629 2932 32e1 e112 1204 0411 117e 7e50 5018 1802 0200 0091 914d 4d00 0000 0014 1403 0303 0300 0001 0101 0116 1603 0303 0302 026a 6a01 0100 0002 0266 6603 0303 0339 3957 5790 90d6 d6e5 e541 418c 8cf4 4504 0405 05dc dcd0 d062 6200 0000 002b 2b06 061b 1b84 8475 7512 12e8 e8c8 c896 96f2 f2a9 a964 6401 01bb bb12 123d 3d12 1204 0411 117e 7eb6 b629 2935 3556 5650 5010 1000 0085 8563 6398 9800 0000 0016 1603 0303 0300 009b 9b02 0200 0000 0097 9703 0303 03ac ac5b 5b8b 8bfd fd50 50fb fb44 449a 9ad8 d878 78ba bab3 b37d 7dc5 4504 0405 05dc dcd0 d063 6300 0000 002b 2b06 061b 1b83 8375 7512 12e8 e8c8 c896 96f2 f2a9 a964 6401 01bb bb12 123d 3d12 1204 0417 1732 32b6 b629 2935 3556 5650 5010 1000 0085 85e6 e6c8 c800 0000 00f9 f953 53a4 a4cb cb74 74a3 a33a 3aee ee66 661f 1f15 15ac ac0a 0a90 90af af65 6548 4819 1916 16cf cff3 f369 69aa aa7c 7ced 4504 0404 04c0 c0d0 d064 6400 0000 002b 2b06 061c 1c9e 9e75 7512 12e8 e8c8 c896 96f2 f2a9 a964 6401 01bb bb12 123d 3d12 1204 041c 1ce6 e6b6 b629 2935 3556 5650 5018 1800 0085 8548 488a 8a00 0000 00b9 b995 954b 4b2b 2b9d 9d6b 6b80 8098 9890 90f8 f8f2 f21c 1caa aa70 708f 8f12 128d 8dbc bc34 3405 055d 5d26 268a 8afa faaa"  # 注释掉的示例文本
    # print(text_a.split(" "))  # 注释掉的调试信息
    # b = args.tokenizer.tokenize(text_a)  # 注释掉的调试信息：对示例文本进行分词
    # print(len(b))  # 注释掉的调试信息：打印分词长度
    # print(b)  # 注释掉的调试信息：打印分词结果

    # exit()  # 注释掉的退出代码

    # Build classification model.
    model = Classifier(args)  # 构建分类模型

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)  # 加载或初始化参数

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备
    model = model.to(args.device)  # 将模型移动到设备

    if args.train_path is None:  # 如果没有训练数据
        print("No train data, only evaluate..")  # 打印信息
        result = evaluate(args, read_dataset(args, args.dev_path))  # 仅进行评估
        return  # 返回

    # Training phase.
    trainset = read_dataset(args, args.train_path)  # 读取训练数据集
    random.shuffle(trainset)  # 打乱训练数据集
    instances_num = len(trainset)  # 获取训练实例数量
    batch_size = args.batch_size  # 获取批次大小

    src = torch.LongTensor([example[0] for example in trainset])  # 创建训练源数据张量
    tgt = torch.LongTensor([example[1] for example in trainset])  # 创建训练目标数据张量
    seg = torch.LongTensor([example[2] for example in trainset])  # 创建训练分段标识张量
    if args.soft_targets:  # 如果使用软目标
        soft_tgt = torch.FloatTensor([example[3] for example in trainset])  # 创建软目标张量
    else:  # 如果不使用软目标
        soft_tgt = None  # 软目标为None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1  # 计算训练步数

    print("Batch size: ", batch_size)  # 打印批次大小
    print("The number of training instances:", instances_num)  # 打印训练实例数量

    optimizer, scheduler = build_optimizer(args, model)  # 构建优化器和调度器

    if args.fp16:  # 如果使用混合精度训练
        try:  # 尝试导入apex
            from apex import amp  # 导入apex混合精度训练库
        except ImportError:  # 如果导入失败
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")  # 抛出错误
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)  # 初始化混合精度训练
        args.amp = amp  # 保存amp对象

    if torch.cuda.device_count() > 1:  # 如果使用多个GPU
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))  # 打印GPU数量
        model = torch.nn.DataParallel(model)  # 使用数据并行
    args.model = model  # 保存模型到参数

    total_loss, result, best_result = 0.0, 0.0, 0.0  # 初始化损失和结果变量

    print("Start training.")  # 打印开始训练信息

    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):  # 遍历训练轮数（带进度条）
        model.train()  # 设置模型为训练模式
        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(
                batch_loader(batch_size, src, tgt, seg, soft_tgt)):  # 遍历训练批次
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch,
                               soft_tgt_batch)  # 训练模型并获取损失
            total_loss += loss.item()  # 累加损失
            if (i + 1) % args.report_steps == 0:  # 如果达到报告步数
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                                  total_loss / args.report_steps))  # 打印训练信息
                total_loss = 0.0  # 重置损失

        result = evaluate(args, read_dataset(args, args.dev_path))  # 在验证集上评估模型
        if result > best_result:  # 如果当前结果优于最佳结果
            best_result = result  # 更新最佳结果
            save_model(model, args.output_model_path)  # 保存模型

    # Evaluation phase.
    if args.test_path is not None:  # 如果有测试路径
        print("Test set evaluation.")  # 打印测试集评估信息
        if torch.cuda.device_count() > 1:  # 如果使用多个GPU
            model.module.load_state_dict(torch.load(args.output_model_path))  # 加载最佳模型（多GPU情况）
        else:  # 如果使用单个GPU或CPU
            model.load_state_dict(torch.load(args.output_model_path))  # 加载最佳模型
        evaluate(args, read_dataset(args, args.test_path), False)  # 在测试集上评估模型


if __name__ == "__main__":  # 如果是主程序
    main()  # 调用主函数
