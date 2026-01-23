import torch.nn as nn
from uer.layers.layer_norm import LayerNorm, T5LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward, GatedFeedForward, FeedForward
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.relative_position_embedding import RelativePositionEmbedding
from uer.layers.moe_layer import MoELayer

class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, args):
        super(TransformerLayer, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale = with_scale
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # Feed forward layer.
        if args.feed_forward == "gated":
            self.feed_forward = GatedFeedForward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
            )
        else:
            self.feed_forward = PositionwiseFeedForward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
            )
        self.dropout_2 = nn.Dropout(args.dropout)

        if args.layernorm == "t5":
            self.layer_norm_1 = T5LayerNorm(args.hidden_size)
            self.layer_norm_2 = T5LayerNorm(args.hidden_size)
        else:
            self.layer_norm_1 = LayerNorm(args.hidden_size)
            self.layer_norm_2 = LayerNorm(args.hidden_size)



    def forward(self, hidden, mask, position_bias = None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        if self.layernorm_positioning == "post":
            inter,probs = self.self_attn(hidden, hidden, hidden, mask, position_bias)
            inter = self.dropout_1(inter)
            inter = self.layer_norm_1(inter + hidden)
            output = self.dropout_2(self.feed_forward(inter))
            output = self.layer_norm_2(output + inter)
        else:
            inter = self.layer_norm_1(hidden)
            inter,probs= self.self_attn(inter, inter, inter, mask, position_bias)
            inter = self.dropout_1(inter)
            hidden = hidden + inter
            output = self.layer_norm_2(hidden)
            output = self.dropout_2(self.feed_forward(output)) + hidden
        return output,probs


class TransformerMOELayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """

    def __init__(self, args):
        # 调用父类 nn.Module 的初始化函数，这是 PyTorch 自定义模块的标准写法
        super(TransformerMOELayer, self).__init__()

        # 从参数中获取 LayerNorm 的位置配置（例如是 pre-norm 还是 post-norm，不过下面的 forward 写法主要支持 post-norm 逻辑）
        self.layernorm_positioning = args.layernorm_positioning

        # 确定注意力头的大小 (Dimension of each attention head)
        if hasattr(args, "attention_head_size"):
            # 如果参数里明确指定了 head size，就直接用
            attention_head_size = args.attention_head_size
        else:
            # 否则通过 隐藏层大小 / 头数 来自动计算
            attention_head_size = args.hidden_size // args.heads_num

        # 确定是否使用偏置项 (Bias)，默认为 True，除非参数里指定移除
        has_bias = bool(1 - args.remove_transformer_bias)
        # 确定 Attention 计算时是否进行缩放 (Scale)，默认为 True
        with_scale = bool(1 - args.remove_attention_scale)

        # Multi-headed self-attention.
        # 实例化多头自注意力层
        # 输入参数：隐藏层大小、头数、每个头的大小、Dropout比率、是否有偏置、是否缩放
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias,
            with_scale=with_scale
        )

        # 定义第一个 Dropout 层，用于 Attention 输出之后
        self.dropout_1 = nn.Dropout(args.dropout)

        # Feed forward layer.
        # === 核心 MoE 部分开始 ===

        # 1. 首先创建一个“专家原型” (ffn)
        # 这是一个标准的 FeedForward 网络 (两层线性层 + LayerNorm)，定义在 uer/layers/position_ffn.py 中
        # args.moebert_expert_dim 定义了专家内部中间层的维度 (通常比 hidden_size 大，例如 4倍)
        ffn = FeedForward(args.hidden_size, args.moebert_expert_dim, args.hidden_act, args.dropout)

        # 2. 实例化 MoELayer 包装器
        # 这个类会接收上面的 ffn 原型，并将其复制 num_experts 份
        self.experts = MoELayer(
            hidden_size=args.hidden_size,  # 输入维度
            expert=ffn,  # 传入刚才创建的专家对象，MoELayer 内部会做 deepcopy
            num_experts=args.moebert_expert_num,  # 专家的数量 (由命令行参数决定，例如 4)
            route_method=args.moebert_route_method,  # 路由方法 (例如 "proto" 表示基于协议ID路由)
            vocab_size=args.vocab_size,  # 词表大小
            hash_list=args.moebert_route_hash_list,  # 路由哈希表
        )
        # === 核心 MoE 部分结束 ===

        # 定义 Layer Normalization 层
        # 支持 T5 风格的 LayerNorm 或 标准 LayerNorm
        if args.layernorm == "t5":
            self.layer_norm_1 = T5LayerNorm(args.hidden_size)  # Attention 后的归一化
            self.layer_norm_2 = T5LayerNorm(args.hidden_size)  # FFN (MoE) 后的归一化 (实际上 MoE 内部可能已经做了，这里可能是为了兼容性或双重保障)
        else:
            self.layer_norm_1 = LayerNorm(args.hidden_size)
            self.layer_norm_2 = LayerNorm(args.hidden_size)

    def forward(self, hidden, mask, position_bias=None, expert_input_ids=None, proto=None):
        """
        Args:
            hidden: 输入的隐藏层状态 [batch_size x seq_length x emb_size]
            mask: 注意力掩码 [batch_size x 1 x seq_length x seq_length] (用于屏蔽 Padding 等)
            position_bias: 相对位置偏置 (可选)
            expert_input_ids: (某些路由方法可能需要 Token ID，但在 proto 模式下可能没用到)
            proto: 关键参数！协议类型 ID 列表 (例如 [0, 1, 0...])，用于指导 MoE 路由
        Returns:
            output: 该层的输出 [batch_size x seq_length x hidden_size]
            balance_loss: 负载均衡损失 (用于训练路由门控，使其不要只盯着一个专家)
        """

        # 1. 计算多头自注意力 (Self-Attention)
        # 输入：Q, K, V 都是 hidden (自己查自己)，以及 mask 和 position_bias
        # 输出：inter (注意力后的结果)
        # self.dropout_1 对结果进行 dropout 处理
        inter = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask, position_bias))

        # 2. Add & Norm (残差连接 + 归一化)
        # 将 Attention 的输出 inter 与原始输入 hidden 相加，然后过 LayerNorm
        inter = self.layer_norm_1(inter + hidden)

        # 3. MoE 专家计算 (Feed Forward 替代品)
        # 调用 self.experts (即 MoELayer)
        # 关键点：将 proto 参数传入。在 "proto" 模式下，MoELayer 会根据这个 ID 决定把 inter 发给哪个专家网络。
        # 返回值：
        #   output: 经过选定专家处理后的数据
        #   balance_loss: 专家的负载均衡损失 (如果用的是 gate 路由会计算，proto 路由通常为 0)
        #   gate_load: (这里接收了但没返回) 每个专家的负载统计
        output, balance_loss, gate_load = self.experts(inter, expert_input_ids, mask, proto=proto)

        # 返回最终输出和负载均衡损失
        # 注意：这里没有像普通 Transformer 那样显式写 output = output + inter 的残差连接代码
        # 原因：在 uer/layers/position_ffn.py 的 FeedForward 类中，最后一行已经写了 self.LayerNorm(hidden_states + input_tensor)
        # 所以专家内部已经完成了 "Add & Norm" 的操作。
        return output, balance_loss

class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super(TransformerDecoderLayer, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale = with_scale
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # Multi-headed context-attention.
        self.context_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale = with_scale
        )
        self.dropout_2 = nn.Dropout(args.dropout)

        # Feed forward layer.
        if args.feed_forward == "gated":
            self.feed_forward = GatedFeedForward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
            )
        else:
            self.feed_forward = PositionwiseFeedForward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
            )
        self.dropout_3 = nn.Dropout(args.dropout)

        # Layer Normalization
        if  args.layernorm == "t5":
            self.layer_norm_1 = T5LayerNorm(args.hidden_size)
            self.layer_norm_2 = T5LayerNorm(args.hidden_size)
            self.layer_norm_3 = T5LayerNorm(args.hidden_size)
        else:
            self.layer_norm_1 = LayerNorm(args.hidden_size)
            self.layer_norm_2 = LayerNorm(args.hidden_size)
            self.layer_norm_3 = LayerNorm(args.hidden_size)



    def forward(self, hidden, encoder_hidden, mask_decoder, mask_encoder, self_position_bias = None, context_position_bias = None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            encoder_hidden: [batch_size x seq_length x emb_size]
            mask_encoder: [batch_size x 1 x seq_length x seq_length]
            mask_decoder: [batch_size x 1 x seq_length x seq_length]
            self_position_bias: [1 x heads_num x seq_length x seq_length]
            context_position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        if self.layernorm_positioning == "post":
            query = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask_decoder, self_position_bias))
            query_norm = self.layer_norm_1(query + hidden)
            mid = self.dropout_2(self.context_attn(encoder_hidden, encoder_hidden, query_norm, mask_encoder, context_position_bias))
            mid_norm = self.layer_norm_2(mid + query_norm)
            output = self.dropout_3(self.feed_forward(mid_norm))
            output = self.layer_norm_3(output + mid_norm)
        else:
            hidden_norm = self.layer_norm_1(hidden)
            query = self.dropout_1(self.self_attn(hidden_norm, hidden_norm, hidden_norm, mask_decoder, self_position_bias))
            query = query + hidden
            query_norm = self.layer_norm_2(query)
            mid = self.dropout_2(self.context_attn(encoder_hidden, encoder_hidden, query_norm, mask_encoder, context_position_bias))
            mid = mid + query
            mid_norm = self.layer_norm_3(mid)
            output = self.dropout_3(self.feed_forward(mid_norm)) + mid
        return output


#class GptBlock(nn.Module):
#    def __init__(self, args):
#        super(GptBlock, self).__init__()

#        # Multi-headed self-attention.
#        self.self_attn = MultiHeadedAttention(
#            args.hidden_size, args.heads_num, args.dropout
#        )
#        self.layer_norm_1 = LayerNorm(args.hidden_size)
#        # Feed forward layer.
#        self.feed_forward = PositionwiseFeedForward(
#            args.hidden_size, args.feedforward_size, args.hidden_act
#        )
#        self.layer_norm_2 = LayerNorm(args.hidden_size)

#    def forward(self, hidden, mask):
#        """
#        Args:
#            hidden: [batch_size x seq_length x emb_size]
#            mask: [batch_size x 1 x seq_length x seq_length]
#        Returns:
#            output: [batch_size x seq_length x hidden_size]
#        """
#        inter = self.layer_norm_1(hidden)
#        inter = self.self_attn(inter, inter, inter, mask)
#        hidden = hidden + inter
#        output = self.layer_norm_2(hidden)
#        output = self.feed_forward(output)
        
#        return output + hidden
