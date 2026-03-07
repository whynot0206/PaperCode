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
import tqdm  # 导入进度条显示模块
import numpy as np  # 导入数值计算库
from sklearn.metrics import f1_score, precision_score, recall_score  # 从sklearn导入评估指标
from uer.macro_moe.encoder import MacroMoEEncoder


class Classifier(nn.Module):  # 定义分类器类，继承自nn.Module
    def __init__(self, args):  # 初始化函数
        super(Classifier, self).__init__()  # 调用父类初始化函数
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))  # 根据参数创建嵌入层
        if args.encoder == "macro_moe":
            self.encoder = MacroMoEEncoder(args)
        else:
            self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num  # 设置标签数量
        self.pooling = args.pooling  # 设置池化方式
        self.soft_targets = args.soft_targets  # 设置是否使用软目标
        self.soft_alpha = args.soft_alpha  # 设置软目标权重
        self.macro_load_balance = getattr(args, "macro_load_balance", 0.1)
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)  # 创建第一个输出层
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)  # 创建第二个输出层（分类层）

    def forward(self, src, tgt, seg, soft_tgt=None):  # 前向传播函数
        """
        Args:
            src: [batch_size x seq_length]  # 输入序列
            tgt: [batch_size]  # 目标标签
            seg: [batch_size x seq_length]  # 分段标识
        """
        # Embedding.
        emb = self.embedding(src, seg)  # 通过嵌入层获取嵌入表示
        # Encoder.
        if hasattr(self, "encoder") and type(self.encoder).__name__ == "MacroMoEEncoder":
            # 接收多出来的 expert_indices
            output, gate_loss, expert_indices = self.encoder(emb, seg)
        else:
            output = self.encoder(emb, seg)
            gate_loss = 0.0
            expert_indices = None  # 如果不是 MoE，就返回 None

        temp_output = output  # 保存临时输出（用于调试或其他用途）
        # Target.
        # Target.
        if self.pooling == "mean":  # 如果使用均值池化
            output = torch.mean(output, dim=1)  # 沿序列维度取均值
        elif self.pooling == "max":  # 如果使用最大池化
            output = torch.max(output, dim=1)[0]  # 沿序列维度取最大值
        elif self.pooling == "last":  # 如果使用最后位置池化
            output = output[:, -1, :]  # 取序列最后一个位置的输出
        else:  # 默认使用第一个位置池化（[CLS]标记）
            output = output[:, 0, :]  # 取序列第一个位置的输出
        output = torch.tanh(self.output_layer_1(output))  # 通过第一个输出层并使用tanh激活函数
        logits = self.output_layer_2(output)  # 通过第二个输出层获取分类logits
        if tgt is not None:  # 如果有目标标签（训练模式）
            if self.soft_targets and soft_tgt is not None:  # 如果使用软目标且提供了软目标
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits),
                                                            tgt.view(-1))  # 计算混合损失（MSE + NLL）
            else:  # 如果不使用软目标
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))  # 计算负对数似然损失

            if isinstance(self.encoder, MacroMoEEncoder) and isinstance(gate_loss, torch.Tensor):
                loss = loss + self.macro_load_balance * gate_loss

            return loss, logits, expert_indices  # 返回损失和logits
        else:  # 如果没有目标标签（预测模式）
            return None, logits, expert_indices  # 返回None和logits
            # return temp_output, logits  # 注释掉的代码：返回临时输出和logits


def count_labels_num(path):  # 计算标签数量的函数
    labels_set, columns = set(), {}  # 初始化标签集合和列字典
    with open(path, mode="r", encoding="utf-8") as f:  # 打开文件
        for line_id, line in enumerate(f):  # 遍历文件行
            if line_id == 0:  # 如果是第一行（表头）
                for i, column_name in enumerate(line.strip().split("\t")):  # 遍历列名
                    columns[column_name] = i  # 记录列索引
                continue  # 继续下一行
            line = line.strip().split("\t")  # 分割行数据
            label = int(line[columns["label"]])  # 获取标签值
            labels_set.add(label)  # 添加到标签集合
    return len(labels_set)  # 返回标签数量


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


def read_dataset(args, path):  # 读取数据集的函数
    dataset, columns = [], {}  # 初始化数据集和列字典
    with open(path, mode="r", encoding="utf-8") as f:  # 打开文件
        for line_id, line in enumerate(f):  # 遍历文件行
            if line_id == 0:  # 如果是第一行（表头）
                for i, column_name in enumerate(line.strip().split("\t")):  # 遍历列名
                    columns[column_name] = i  # 记录列索引
                continue  # 继续下一行
            line = line[:-1].split("\t")  # 分割行数据（去除换行符）
            tgt = int(line[columns["label"]])  # 获取标签
            if args.soft_targets and "logits" in columns.keys():  # 如果使用软目标且数据中有logits列
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]  # 获取软目标
            if "text_b" not in columns:  # 如果没有text_b列（单句分类）
                text_a = line[columns["text_a"]]  # 获取文本a
                src = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + args.tokenizer.tokenize(text_a))  # 将文本转换为ID序列（添加[CLS]）
                seg = [1] * len(src)  # 创建分段标识（全为1）
            else:  # 如果有text_b列（句对分类）
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]  # 获取文本a和文本b
                src_a = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])  # 将文本a转换为ID序列
                src_b = args.tokenizer.convert_tokens_to_ids(
                    args.tokenizer.tokenize(text_b) + [SEP_TOKEN])  # 将文本b转换为ID序列
                src = src_a + src_b  # 拼接两个序列
                seg = [1] * len(src_a) + [2] * len(src_b)  # 创建分段标识（文本a为1，文本b为2）

            if len(src) > args.seq_length:  # 如果序列长度超过最大长度
                src = src[: args.seq_length]  # 截断序列
                seg = seg[: args.seq_length]  # 截断分段标识
            while len(src) < args.seq_length:  # 如果序列长度小于最大长度
                src.append(0)  # 填充0
                seg.append(0)  # 填充0
            if args.soft_targets and "logits" in columns.keys():  # 如果使用软目标且数据中有logits列
                dataset.append((src, tgt, seg, soft_tgt))  # 添加数据（含软目标）
            else:  # 如果不使用软目标或数据中没有logits列
                dataset.append((src, tgt, seg))  # 添加数据（不含软目标）

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
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)  # 初始化混淆矩阵
    y_true, y_pred = [], []  # 初始化真实标签和预测标签列表
    all_expert_indices = []  # <--- 新增：用于收集所有的专家分配结果
    args.model.eval()  # 设置模型为评估模式

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):  # 遍历批次数据
        src_batch = src_batch.to(args.device)  # 将源数据批次移动到设备
        # print(src_batch[0][113],args.tokenizer.convert_ids_to_tokens([src_batch.cpu().numpy()[0][113]]))  # 注释掉的调试信息
        tgt_batch = tgt_batch.to(args.device)  # 将目标数据批次移动到设备
        seg_batch = seg_batch.to(args.device)  # 将分段标识批次移动到设备
        with torch.no_grad():  # 禁用梯度计算
            _, logits, expert_indices = args.model(src_batch, tgt_batch, seg_batch)  # 前向传播获取logits
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)  # 获取预测结果（取最大概率的类别）
        gold = tgt_batch  # 获取真实标签
        for j in range(pred.size()[0]):  # 遍历批次中的每个样本
            confusion[pred[j], gold[j]] += 1  # 更新混淆矩阵
            y_true.append(gold[j].cpu())  # 添加真实标签到列表
            y_pred.append(pred[j].cpu())  # 添加预测标签到列表

            # 新增：把这个样本的路由结果存下来
            if expert_indices is not None:
                all_expert_indices.append(expert_indices[j].cpu().item())

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

        if len(all_expert_indices) > 0:
            import json
            routing_data = {
                "true_labels": [int(y) for y in y_true],
                "expert_indices": all_expert_indices
            }
            with open("routing_analysis.json", "w") as f:
                json.dump(routing_data, f)  # <--- 直接把数据 dump 给文件对象 f
            print("路由数据已保存：routing_analysis.json!")

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))  # 打印准确率
    print("Macro precision: {:.4f}, Micro precision: {:.4f}, Weighted precision: {:.4f}".format(
        precision_score(y_true, y_pred, average='macro'), precision_score(y_true, y_pred, average='micro'),
        precision_score(y_true, y_pred, average='weighted')))  # 打印各种精确率
    print("Macro recall: {:.4f}, Micro recall: {:.4f}, Weighted recall: {:.4f}".format(
        recall_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='micro'),
        recall_score(y_true, y_pred, average='weighted')))  # 打印各种召回率
    print("Macro f1: {:.4f}, Micro f1: {:.4f}, Weighted f1: {:.4f}".format(
        f1_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='micro'),
        f1_score(y_true, y_pred, average='weighted')))  # 打印各种F1分数

    return f1_score(y_true, y_pred, average='macro'), confusion  # 返回宏F1分数和混淆矩阵


def main():  # 主函数
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 创建参数解析器

    finetune_opts(parser)  # 添加微调选项

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")  # 添加池化方式参数

    parser.add_argument("--earlystop", type=int, default=5, help="early stop rounds.")  # 添加早停参数

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                        )  # 添加分词器参数

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")  # 添加软目标参数
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")  # 添加软目标权重参数

    # MOE Model Options
    parser.add_argument("--is_moe", action="store_true", help="adopt moe layer.")  # 添加是否使用MOE层参数
    parser.add_argument("--vocab_size", type=int, required=False, help="Number of vocab.")  # 添加词汇表大小参数
    parser.add_argument("--moebert_expert_dim", type=int, required=False, default=3072,
                        help="Dim of expert,default is ffn.")  # 添加MOE专家维度参数
    parser.add_argument("--moebert_expert_num", type=int, required=False, help="Number of expert.")  # 添加MOE专家数量参数
    parser.add_argument("--moebert_route_method",
                        choices=["gate-token", "gate-sentence", "hash-random", "hash-balance", "proto"],
                        default="hash-random",
                        help="moebert route method.")  # 添加MOE路由方法参数
    parser.add_argument("--moebert_route_hash_list", default=None, type=str,
                        help="Path of moebert hash list file.")  # 添加MOE哈希列表路径参数
    parser.add_argument("--moebert_load_balance", type=float, default=0.1, help="gate loss weight.")  # 添加MOE负载平衡参数

    args = parser.parse_args()  # 解析参数

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)  # 从配置文件加载超参数

    set_seed(args.seed)  # 设置随机种子

    # Count the number of labels.
    if args.train_path is None:  # 如果没有训练路径
        args.labels_num = 197  # 设置默认标签数量
    else:  # 如果有训练路径
        args.labels_num = count_labels_num(args.train_path)  # 计算标签数量

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)  # 构建分词器

    # Build classification model.
    model = Classifier(args)  # 构建分类模型

    # 动态判断是否开启 Few-shot 适配模式
    if args.encoder == "macro_moe":
        # 如果命令里传了 --few_shot_stage
        if hasattr(args, "few_shot_stage") and args.few_shot_stage:
            print("Enable Few-shot Adaptation Mode: Freezing Backbone, Training Adapters.")
            model.encoder.set_adaptation_mode(True)
        else:
            print("Enable Full Fine-tuning Mode: Unfreezing Backbone.")
            model.encoder.set_adaptation_mode(False)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)  # 加载或初始化参数

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备
    model = model.to(args.device)  # 将模型移动到设备

    if args.train_path is None:  # 如果没有训练数据
        args.model = model  # 设置模型参数
        args.labels_num = 197  # 设置标签数量
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
    best_result_round = 0  # 初始化最佳结果轮数
    # print("Start training.")  # 注释掉的开始训练信息

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
        if result[0] > best_result:  # 如果当前结果优于最佳结果
            best_result = result[0]  # 更新最佳结果
            best_result_round = epoch  # 更新最佳结果轮数
            save_model(model, args.output_model_path)  # 保存模型
        elif epoch - best_result_round >= args.earlystop:  # 如果达到早停条件
            print("early stopping...")  # 打印早停信息
            break  # 跳出训练循环

    # Evaluation phase.
    if args.test_path is not None:  # 如果有测试路径
        print("Test set evaluation.")  # 打印测试集评估信息
        if torch.cuda.device_count() > 1:  # 如果使用多个GPU
            model.module.load_state_dict(torch.load(args.output_model_path))  # 加载最佳模型（多GPU情况）
        else:  # 如果使用单个GPU或CPU
            model.load_state_dict(torch.load(args.output_model_path))  # 加载最佳模型
        evaluate(args, read_dataset(args, args.test_path), True)  # 在测试集上评估模型并打印混淆矩阵


if __name__ == "__main__":  # 如果是主程序
    main()  # 调用主函数
