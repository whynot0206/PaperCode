import time  # 导入时间模块，用于计时和延迟
import torch  # 导入PyTorch深度学习框架
import torch.distributed as dist  # 导入PyTorch分布式训练模块
import torch.multiprocessing as mp  # 导入PyTorch多进程模块
from torch.nn.parallel import DistributedDataParallel  # 导入分布式数据并行包装器
from uer.model_loader import load_model  # 从UER导入模型加载函数
from uer.model_saver import save_model  # 从UER导入模型保存函数
from uer.model_builder import build_model  # 从UER导入模型构建函数
from uer.utils.optimizers import *  # 从UER工具包导入所有优化器
from uer.utils import *  # 从UER工具包导入所有工具函数
from uer.utils.vocab import Vocab  # 从UER工具包导入词汇表类
from uer.utils.seed import set_seed  # 从UER工具包导入设置随机种子函数
from tqdm import tqdm  # 导入进度条显示库


def train_and_validate(args):  # 定义训练和验证的主函数，接收参数args
    set_seed(args.seed)  # 设置随机种子以确保实验可重复

    # Load vocabulary.  # 加载词汇表
    print("Load vocabulary.")  # 打印提示信息
    if args.spm_model_path:  # 如果提供了SentencePiece模型路径
        try:  # 尝试导入sentencepiece库
            import sentencepiece as spm  # 导入sentencepiece库
        except ImportError:  # 如果导入失败
            raise ImportError(  # 抛出导入错误异常
                "You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece")  # 提示用户安装sentencepiece
        sp_model = spm.SentencePieceProcessor()  # 创建SentencePiece处理器实例
        sp_model.Load(args.spm_model_path)  # 加载预训练的SentencePiece模型
        args.vocab = {sp_model.IdToPiece(i): i for i  # 构建词汇表字典，将id映射到token
                      in range(sp_model.GetPieceSize())}  # 遍历所有piece
        args.tokenizer = str2tokenizer[args.tokenizer](args)  # 根据名称创建对应的tokenizer实例
        if args.target == "seq2seq":  # 如果是seq2seq任务
            tgt_sp_model = spm.SentencePieceProcessor()  # 创建目标语言的SentencePiece处理器
            tgt_sp_model.Load(args.tgt_spm_model_path)  # 加载目标语言的SentencePiece模型
            args.tgt_vocab = {tgt_sp_model.IdToPiece(i): i for i  # 构建目标语言词汇表字典
                              in range(tgt_sp_model.GetPieceSize())}  # 遍历所有piece
    else:  # 如果没有提供SentencePiece模型路径
        args.tokenizer = str2tokenizer[args.tokenizer](args)  # 根据名称创建对应的tokenizer实例
        args.vocab = args.tokenizer.vocab  # 从tokenizer获取词汇表
        if args.target == "seq2seq":  # 如果是seq2seq任务
            tgt_vocab = Vocab()  # 创建目标语言词汇表实例
            tgt_vocab.load(args.tgt_vocab_path)  # 加载目标语言词汇表文件
            args.tgt_vocab = tgt_vocab.w2i  # 获取目标语言的词到索引映射

    # Build model.  # 构建模型
    print("Build model.")  # 打印提示信息
    model = build_model(args)  # 根据参数构建模型
    # for name,parameters in model.named_parameters():  # 注释掉的代码：打印所有参数名称和大小
    #     print(name,':',parameters.size())  # 打印参数名称和大小

    # Load or initialize parameters.  # 加载或初始化参数
    if args.pretrained_model_path is not None:  # 如果提供了预训练模型路径
        # Initialize with pretrained model.  # 使用预训练模型初始化
        model = load_model(model, args.pretrained_model_path)  # 加载预训练模型参数
    else:  # 如果没有提供预训练模型
        # Initialize with normal distribution.  # 使用正态分布初始化
        for n, p in list(model.named_parameters()):  # 遍历所有模型参数
            if "gamma" not in n and "beta" not in n:  # 如果不是gamma或beta参数（LayerNorm参数）
                p.data.normal_(0, 0.02)  # 使用正态分布初始化参数

    if args.dist_train:  # 如果启用分布式训练
        # Multiprocessing distributed mode.  # 多进程分布式模式
        print("Multiprocessing distributed mode.")  # 打印提示信息
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model), daemon=False)  # 启动多进程训练
    elif args.single_gpu:  # 如果使用单GPU训练
        # Single GPU mode.  # 单GPU模式
        print("Single GPU mode.")  # 打印提示信息
        worker(args.gpu_id, None, args, model)  # 调用worker函数进行单GPU训练
    else:  # 否则使用CPU训练
        # CPU mode.  # CPU模式
        print("CPU mode.")  # 打印提示信息
        worker(None, None, args, model)  # 调用worker函数进行CPU训练


class Trainer(object):  # 定义训练器基类
    def __init__(self, args):  # 初始化训练器
        self.current_step = 1  # 初始化当前训练步数为1
        self.total_steps = args.total_steps  # 设置总训练步数
        self.accumulation_steps = args.accumulation_steps  # 设置梯度累积步数
        self.report_steps = args.report_steps  # 设置报告间隔步数
        self.save_checkpoint_steps = args.save_checkpoint_steps  # 设置保存检查点间隔步数

        self.output_model_path = args.output_model_path  # 设置模型输出路径

        self.start_time = time.time()  # 记录训练开始时间
        self.total_loss = 0.0  # 初始化总损失值

        self.dist_train = args.dist_train  # 设置是否分布式训练标志
        self.batch_size = args.batch_size  # 设置批大小
        self.world_size = args.world_size  # 设置世界大小（进程数）

    def forward_propagation(self, batch, model):  # 定义前向传播方法（需子类实现）
        raise NotImplementedError  # 抛出未实现错误

    def report_and_reset_stats(self):  # 定义报告和重置统计信息方法（需子类实现）
        raise NotImplementedError  # 抛出未实现错误

    def train(self, args, gpu_id, rank, loader, model, optimizer, scheduler):  # 定义训练循环
        model.train()  # 设置模型为训练模式
        loader_iter = iter(loader)  # 创建数据加载器的迭代器

        while True:  # 开始训练循环
            if self.current_step == self.total_steps + 1:  # 如果达到总训练步数+1
                break  # 退出训练循环
            batch = list(next(loader_iter))  # 获取下一个批次数据并转换为列表
            self.seq_length = batch[0].size(1)  # 记录序列长度（从第一个张量的第二维度获取）
            if gpu_id is not None:  # 如果使用了GPU
                for i in range(len(batch)):  # 遍历批次中的所有张量
                    batch[i] = batch[i].cuda(gpu_id)  # 将张量移动到指定GPU

            loss = self.forward_propagation(batch, model)  # 执行前向传播并计算损失

            if args.fp16:  # 如果使用混合精度训练
                with args.amp.scale_loss(loss, optimizer) as scaled_loss:  # 使用AMP缩放损失
                    scaled_loss.backward()  # 执行反向传播
            else:  # 如果不使用混合精度训练
                loss.backward()  # 执行反向传播

            if self.current_step % self.accumulation_steps == 0:  # 如果达到梯度累积步数
                optimizer.step()  # 更新模型参数
                scheduler.step()  # 更新学习率
                model.zero_grad()  # 清空梯度

            # 如果达到报告间隔步数且
            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):  # （非分布式或分布式且rank为0）

                # ====== DDP 兼容：统一获取 raw model ======
                raw_model = model.module if hasattr(model, "module") else model

                # ====== 调试信息：显示当前模型类型 ======
                print(f"  [DEBUG] Current model type: {type(raw_model).__name__}")
                if hasattr(raw_model, "encoder"):
                    print(f"  [DEBUG] Encoder type: {type(raw_model.encoder).__name__}")
                else:
                    print(f"  [DEBUG] No encoder found in model")

                # ====== 打印 Macro-MoE Router 使用情况 ======
                router_found = False
                if hasattr(raw_model, "encoder"):
                    if hasattr(raw_model.encoder, "router"):
                        router = raw_model.encoder.router
                        usage = router.usage_counter.cpu().tolist()
                        total = sum(usage) + 1e-6
                        usage_ratio = [u / total for u in usage]

                        print("  [MoE Router Usage] count =", usage)
                        print("  [MoE Router Usage] ratio =", ["{:.3f}".format(r) for r in usage_ratio])

                        # ====== 打印后立刻 reset，形成滑动窗口统计 ======
                        router.reset_usage()
                        router_found = True
                    else:
                        print(f"  [DEBUG] Encoder has no router attribute")

                if not router_found:
                    print("  [INFO] Current model is not MacroMoE, skipping router usage stats")

                # ====== 打印 Expert Backbone 梯度范数 ======
                experts_found = False
                if hasattr(raw_model, "encoder"):
                    if hasattr(raw_model.encoder, "experts"):
                        print("  [MoE Expert Grad Norms]")
                        for i, expert in enumerate(raw_model.encoder.experts):
                            grad_norm = expert.get_backbone_grad_norm()
                            print(f"    Expert-{i}: grad_norm = {grad_norm:.6f}")
                        experts_found = True
                    else:
                        print(f"  [DEBUG] Encoder has no experts attribute")

                if not experts_found:
                    print("  [INFO] Current model has no experts, skipping grad norm stats")

                self.report_and_reset_stats()  # 报告并重置统计信息
                self.start_time = time.time()  # 重置开始时间

            # 如果达到保存检查点间隔步数且
            if self.current_step % self.save_checkpoint_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):  # （非分布式或分布式且rank为0）
                save_model(model, self.output_model_path + "-" + str(self.current_step))  # 保存模型检查点

            self.current_step += 1  # 更新当前训练步数


class MlmTrainer(Trainer):  # 定义MLM任务训练器，继承自Trainer
    def __init__(self, args):  # 初始化MLM训练器
        super(MlmTrainer, self).__init__(args)  # 调用父类初始化方法
        self.total_correct = 0.0  # 初始化总正确预测数
        self.total_denominator = 0.0  # 初始化总分母数（总掩码token数）

    def forward_propagation(self, batch, model):  # 定义MLM前向传播
        src, tgt, seg = batch  # 解包批次数据：输入序列、目标序列、段落标记
        loss_info = model(src, tgt, seg)  # 调用模型前向传播，返回损失信息
        loss, correct, denominator = loss_info  # 解包损失信息：总损失、正确预测数、分母数
        self.total_loss += loss.item()  # 累加总损失
        self.total_correct += correct.item()  # 累加总正确预测数
        self.total_denominator += denominator.item()  # 累加总分母数
        loss = loss / self.accumulation_steps  # 根据梯度累积步数缩放损失
        return loss  # 返回缩放后的损失

    def report_and_reset_stats(self):  # 定义MLM报告和重置统计信息
        done_tokens = self.batch_size * self.seq_length * self.report_steps  # 计算已处理的token数
        if self.dist_train:  # 如果是分布式训练
            done_tokens *= self.world_size  # 乘以世界大小得到总token数
        print("| {:8d}/{:8d} steps"  # 打印训练进度信息
              "| {:8.2f} tokens/s"  # 打印处理速度
              "| loss {:7.2f}"  # 打印平均损失
              "| acc: {:3.3f}".format(  # 打印准确率
            self.current_step,  # 当前步数
            self.total_steps,  # 总步数
            done_tokens / (time.time() - self.start_time),  # 计算每秒处理的token数
            self.total_loss / self.report_steps,  # 计算平均损失
            self.total_correct / self.total_denominator))  # 计算准确率

        self.total_loss = 0.0  # 重置总损失
        self.total_correct = 0.0  # 重置总正确预测数
        self.total_denominator = 0.0  # 重置总分母数


class BertTrainer(Trainer):  # 定义BERT任务训练器，继承自Trainer
    def __init__(self, args):  # 初始化BERT训练器
        super(BertTrainer, self).__init__(args)  # 调用父类初始化方法
        self.total_loss_sp = 0.0  # 初始化句子对预测任务总损失
        self.total_correct_sp = 0.0  # 初始化句子对预测任务总正确预测数
        self.total_instances = 0.0  # 初始化总实例数

        self.total_loss_mlm = 0.0  # 初始化MLM任务总损失
        self.total_correct_mlm = 0.0  # 初始化MLM任务总正确预测数
        self.total_denominator = 0.0  # 初始化MLM任务总分母数
        self.load_balance_alpha = args.moebert_load_balance  # 设置MoE负载均衡系数
        self.is_moe = args.is_moe  # 设置是否使用MoE模型

    def forward_propagation(self, batch, model):  # 定义BERT前向传播
        debug_mode = False  # 设置调试模式为False
        if debug_mode:  # 如果启用调试模式
            print("In function forward_propagation(self, batch, model):")  # 打印调试信息
            print("type of batch:", type(batch))  # 打印batch类型
            print("type of the content of batch:", [type(elem) for elem in batch])  # 打印batch中各元素类型
        if len(batch) == 5:  # 如果batch包含5个元素（包含协议信息）
            src, tgt_mlm, tgt_sp, seg, proto = batch  # 解包批次数据：输入序列、MLM目标、句子对目标、段落标记、协议信息
            loss_info = model(src, (tgt_mlm, tgt_sp), seg, proto)  # 调用模型前向传播（带协议信息）
        else:  # 如果batch不包含协议信息
            src, tgt_mlm, tgt_sp, seg = batch  # 解包批次数据：输入序列、MLM目标、句子对目标、段落标记
            loss_info = model(src, (tgt_mlm, tgt_sp), seg)  # 调用模型前向传播

        if self.is_moe:  # 如果使用MoE模型
            loss_mlm, loss_sp, correct_mlm, correct_sp, denominator, gate_loss = loss_info  # 解包损失信息（含门控损失）
        else:  # 如果不使用MoE模型
            loss_mlm, loss_sp, correct_mlm, correct_sp, denominator = loss_info  # 解包损失信息
            gate_loss = 0.0  # 设置门控损失为0
        loss = loss_mlm / 10 + loss_sp + self.load_balance_alpha * gate_loss  # 计算总损失（MLM损失缩放+句子对损失+门控损失）
        self.total_loss += loss.item()  # 累加总损失
        self.total_loss_mlm += loss_mlm.item()  # 累加MLM损失
        self.total_loss_sp += loss_sp.item()  # 累加句子对损失
        self.total_correct_mlm += correct_mlm.item()  # 累加MLM正确预测数
        self.total_correct_sp += correct_sp.item()  # 累加句子对正确预测数
        self.total_denominator += denominator.item()  # 累加MLM分母数
        self.total_instances += src.size(0)  # 累加总实例数
        loss = loss / self.accumulation_steps  # 根据梯度累积步数缩放损失
        return loss  # 返回缩放后的损失

    def report_and_reset_stats(self):  # 定义BERT报告和重置统计信息
        done_tokens = self.batch_size * self.seq_length * self.report_steps  # 计算已处理的token数
        if self.dist_train:  # 如果是分布式训练
            done_tokens *= self.world_size  # 乘以世界大小得到总token数

        print("| {:8d}/{:8d} steps"  # 打印训练进度信息
              "| {:3.3f} s"  # 打印时间信息
              "| {:8.2f} tokens/s"  # 打印处理速度
              "| loss {:7.2f}"  # 打印总平均损失
              "| loss_mlm: {:3.3f}"  # 打印MLM平均损失
              "| loss_sp: {:3.3f}"  # 打印句子对平均损失
              "| acc_mlm: {:3.3f}"  # 打印MLM准确率
              "| acc_sp: {:3.3f}".format(  # 打印句子对准确率
            self.current_step,  # 当前步数
            self.total_steps,  # 总步数
            (time.time() - self.start_time),  # 计算经过的时间
            done_tokens / (time.time() - self.start_time),  # 计算每秒处理的token数
            self.total_loss / self.report_steps,  # 计算总平均损失
            self.total_loss_mlm / self.report_steps,  # 计算MLM平均损失
            self.total_loss_sp / self.report_steps,  # 计算句子对平均损失
            self.total_correct_mlm / self.total_denominator,  # 计算MLM准确率
            self.total_correct_sp / self.total_instances))  # 计算句子对准确率

        self.total_loss, self.total_loss_mlm, self.total_loss_sp = 0.0, 0.0, 0.0  # 重置所有损失
        self.total_correct_mlm, self.total_denominator = 0.0, 0.0  # 重置MLM统计信息
        self.total_correct_sp, self.total_instances = 0.0, 0.0  # 重置句子对统计信息


class AlbertTrainer(BertTrainer):  # 定义ALBERT任务训练器，继承自BertTrainer
    pass  # 直接复用BERT训练器的所有方法


class LmTrainer(MlmTrainer):  # 定义语言模型训练器，继承自MlmTrainer
    pass  # 直接复用MLM训练器的所有方法


class BilmTrainer(Trainer):  # 定义双向语言模型训练器，继承自Trainer
    def __init__(self, args):  # 初始化BiLM训练器
        super(BilmTrainer, self).__init__(args)  # 调用父类初始化方法
        self.total_loss_forward, self.total_loss_backward = 0.0, 0.0  # 初始化前向和后向损失
        self.total_correct_forward, self.total_correct_backward = 0.0, 0.0  # 初始化前向和后向正确预测数
        self.total_denominator = 0.0  # 初始化总分母数

    def forward_propagation(self, batch, model):  # 定义BiLM前向传播
        src, tgt_forward, tgt_backward, seg = batch  # 解包批次数据：输入序列、前向目标、后向目标、段落标记
        loss_info = model(src, (tgt_forward, tgt_backward), seg)  # 调用模型前向传播
        loss_forward, loss_backward, correct_forward, correct_backward, denominator = loss_info  # 解包损失信息
        loss = loss_forward + loss_backward  # 计算总损失（前向+后向）
        self.total_loss += loss.item()  # 累加总损失
        self.total_loss_forward += loss_forward.item()  # 累加前向损失
        self.total_loss_backward += loss_backward.item()  # 累加后向损失
        self.total_correct_forward += correct_forward.item()  # 累加前向正确预测数
        self.total_correct_backward += correct_backward.item()  # 累加后向正确预测数
        self.total_denominator += denominator.item()  # 累加分母数
        loss = loss / self.accumulation_steps  # 根据梯度累积步数缩放损失
        return loss  # 返回缩放后的损失

    def report_and_reset_stats(self):  # 定义BiLM报告和重置统计信息
        done_tokens = self.batch_size * self.seq_length * self.report_steps  # 计算已处理的token数
        if self.dist_train:  # 如果是分布式训练
            done_tokens *= self.world_size  # 乘以世界大小得到总token数
        print("| {:8d}/{:8d} steps"  # 打印训练进度信息
              "| {:8.2f} tokens/s"  # 打印处理速度
              "| loss {:7.2f}"  # 打印总平均损失
              "| loss_forward {:3.3f}"  # 打印前向平均损失
              "| loss_backward {:3.3f}"  # 打印后向平均损失
              "| acc_forward: {:3.3f}"  # 打印前向准确率
              "| acc_backward: {:3.3f}".format(  # 打印后向准确率
            self.current_step,  # 当前步数
            self.total_steps,  # 总步数
            done_tokens / (time.time() - self.start_time),  # 计算每秒处理的token数
            self.total_loss / self.report_steps,  # 计算总平均损失
            self.total_loss_forward / self.report_steps,  # 计算前向平均损失
            self.total_loss_backward / self.report_steps,  # 计算后向平均损失
            self.total_correct_forward / self.total_denominator,  # 计算前向准确率
            self.total_correct_backward / self.total_denominator))  # 计算后向准确率

        self.total_loss, self.total_loss_forward, self.total_loss_backward = 0.0, 0.0, 0.0  # 重置所有损失
        self.total_correct_forward, self.total_correct_backward, self.total_denominator = 0.0, 0.0, 0.0  # 重置所有统计信息


class ClsTrainer(Trainer):  # 定义分类任务训练器，继承自Trainer
    def __init__(self, args):  # 初始化分类训练器
        super(ClsTrainer, self).__init__(args)  # 调用父类初始化方法
        self.total_correct = 0.0  # 初始化总正确预测数
        self.total_instances = 0.0  # 初始化总实例数

    def forward_propagation(self, batch, model):  # 定义分类前向传播
        src, tgt, seg = batch  # 解包批次数据：输入序列、目标标签、段落标记
        loss_info = model(src, tgt, seg)  # 调用模型前向传播
        loss, correct = loss_info  # 解包损失信息：损失、正确预测数
        self.total_loss += loss.item()  # 累加总损失
        self.total_correct += correct.item()  # 累加总正确预测数
        self.total_instances += src.size(0)  # 累加总实例数
        loss = loss / self.accumulation_steps  # 根据梯度累积步数缩放损失
        return loss  # 返回缩放后的损失

    def report_and_reset_stats(self):  # 定义分类报告和重置统计信息
        done_tokens = self.batch_size * self.seq_length * self.report_steps  # 计算已处理的token数
        if self.dist_train:  # 如果是分布式训练
            done_tokens *= self.world_size  # 乘以世界大小得到总token数
        print("| {:8d}/{:8d} steps"  # 打印训练进度信息
              "| {:8.2f} tokens/s"  # 打印处理速度
              "| loss {:7.2f}"  # 打印平均损失
              "| acc: {:3.3f}".format(  # 打印准确率
            self.current_step,  # 当前步数
            self.total_steps,  # 总步数
            done_tokens / (time.time() - self.start_time),  # 计算每秒处理的token数
            self.total_loss / self.report_steps,  # 计算平均损失
            self.total_correct / self.total_instances))  # 计算准确率

        self.total_loss = 0.0  # 重置总损失
        self.total_correct = 0.0  # 重置总正确预测数
        self.total_instances = 0.0  # 重置总实例数


class Seq2seqTrainer(Trainer):  # 定义序列到序列任务训练器，继承自Trainer
    def __init__(self, args):  # 初始化Seq2Seq训练器
        super(Seq2seqTrainer, self).__init__(args)  # 调用父类初始化方法
        self.total_correct = 0.0  # 初始化总正确预测数
        self.total_denominator = 0.0  # 初始化总分母数

    def forward_propagation(self, batch, model):  # 定义Seq2Seq前向传播
        src, tgt_in, tgt_out, seg = batch  # 解包批次数据：源序列、目标输入序列、目标输出序列、段落标记
        loss_info = model(src, (tgt_in, tgt_out, src), seg)  # 调用模型前向传播
        loss, correct, denominator = loss_info  # 解包损失信息：损失、正确预测数、分母数
        self.total_loss += loss.item()  # 累加总损失
        self.total_correct += correct.item()  # 累加总正确预测数
        self.total_denominator += denominator.item()  # 累加总分母数

        loss = loss / self.accumulation_steps  # 根据梯度累积步数缩放损失

        return loss  # 返回缩放后的损失

    def report_and_reset_stats(self):  # 定义Seq2Seq报告和重置统计信息
        done_tokens = self.batch_size * self.seq_length * self.report_steps  # 计算已处理的token数
        if self.dist_train:  # 如果是分布式训练
            done_tokens *= self.world_size  # 乘以世界大小得到总token数

        print("| {:8d}/{:8d} steps"  # 打印训练进度信息
              "| {:8.2f} tokens/s"  # 打印处理速度
              "| loss {:7.2f}"  # 打印平均损失
              "| acc: {:3.3f}".format(  # 打印准确率
            self.current_step,  # 当前步数
            self.total_steps,  # 总步数
            done_tokens / (time.time() - self.start_time),  # 计算每秒处理的token数
            self.total_loss / self.report_steps,  # 计算平均损失
            self.total_correct / self.total_denominator))  # 计算准确率

        self.total_loss = 0.0  # 重置总损失
        self.total_correct = 0.0  # 重置总正确预测数
        self.total_denominator = 0.0  # 重置总分母数


class T5Trainer(Seq2seqTrainer):  # 定义T5任务训练器，继承自Seq2seqTrainer
    pass  # 直接复用Seq2Seq训练器的所有方法


class PrefixlmTrainer(MlmTrainer):  # 定义前缀语言模型训练器，继承自MlmTrainer
    pass  # 直接复用MLM训练器的所有方法


str2trainer = {"bert": BertTrainer, "bertflow": BertTrainer, "mlm": MlmTrainer, "lm": LmTrainer,
               "albert": AlbertTrainer, "bilm": BilmTrainer, "cls": ClsTrainer,
               "seq2seq": Seq2seqTrainer, "t5": T5Trainer}  # 创建任务名称到训练器的映射字典


def worker(proc_id, gpu_ranks, args, model):  # 定义工作进程函数
    """
    Args:  # 参数说明
        proc_id: The id of GPU for single GPU mode;  # 单GPU模式下的GPU ID
                 The id of process (and GPU) for multiprocessing distributed mode.  # 多进程分布式模式下的进程（和GPU）ID
        gpu_ranks: List of ranks of each process.  # 每个进程的rank列表
    """
    set_seed(args.seed)  # 设置随机种子

    print("Starting worker function")  # 打印工作进程启动信息

    if args.dist_train:  # 如果是分布式训练
        rank = gpu_ranks[proc_id]  # 从gpu_ranks获取当前进程的rank
        gpu_id = proc_id  # 设置GPU ID为进程ID
    elif args.single_gpu:  # 如果是单GPU训练
        rank = None  # rank为None
        gpu_id = proc_id  # 设置GPU ID为进程ID
    else:  # 如果是CPU训练
        rank = None  # rank为None
        gpu_id = None  # GPU ID为None
    print("train loader constructing...")  # 打印数据加载器构建信息
    if args.dist_train:  # 如果是分布式训练
        train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, rank, args.world_size,
                                                   True)  # 创建分布式数据加载器
    else:  # 如果不是分布式训练
        train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, 0, 1, True)  # 创建普通数据加载器

    if gpu_id is not None:  # 如果使用了GPU
        torch.cuda.set_device(gpu_id)  # 设置当前GPU设备
        model.cuda(gpu_id)  # 将模型移动到指定GPU

    # Build optimizer.  # 构建优化器
    print("build optomizer...")  # 打印优化器构建信息
    param_optimizer = list(model.named_parameters())  # 获取所有模型参数名称和值
    no_decay = ["bias", "gamma", "beta"]  # 定义不进行权重衰减的参数名称
    optimizer_grouped_parameters = [  # 分组优化器参数
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay_rate": 0.01},  # 应用权重衰减的参数
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0}  # 不应用权重衰减的参数
    ]
    if args.optimizer in ["adamw"]:  # 如果使用AdamW优化器
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  correct_bias=False)  # 创建AdamW优化器实例
    else:  # 如果使用其他优化器
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)  # 创建其他优化器实例
    if args.scheduler in ["constant"]:  # 如果使用常量学习率调度器
        scheduler = str2scheduler[args.scheduler](optimizer)  # 创建常量学习率调度器实例
    elif args.scheduler in ["constant_with_warmup"]:  # 如果使用带热身的常量学习率调度器
        scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps * args.warmup)  # 创建带热身的常量学习率调度器实例
    else:  # 如果使用其他学习率调度器
        scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps * args.warmup, args.total_steps)  # 创建其他学习率调度器实例

    if args.fp16:  # 如果使用混合精度训练
        try:  # 尝试导入apex库
            from apex import amp  # 导入apex混合精度训练库
        except ImportError:  # 如果导入失败
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")  # 提示安装apex
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)  # 初始化混合精度训练
        args.amp = amp  # 将amp实例保存到args中

    if args.dist_train:  # 如果是分布式训练
        print("Initialize multiprocessing distributed training environment...")  # 打印分布式训练环境初始化信息
        # Initialize multiprocessing distributed training environment.  # 初始化多进程分布式训练环境
        dist.init_process_group(backend=args.backend,  # 初始化进程组，指定后端
                                init_method=args.master_ip,  # 指定主节点地址
                                world_size=args.world_size,  # 指定世界大小
                                rank=rank)  # 指定当前进程rank
        model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)  # 包装模型为分布式数据并行模型
        print("Worker %d is training ... " % rank)  # 打印当前工作进程训练信息
    else:  # 如果不是分布式训练
        print("Worker is training ...")  # 打印工作进程训练信息

    trainer = str2trainer[args.target](args)  # 根据任务名称创建对应的训练器实例
    trainer.train(args, gpu_id, rank, train_loader, model, optimizer, scheduler)  # 开始训练