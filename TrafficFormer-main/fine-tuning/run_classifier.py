"""
This script provides an example to wrap UER-py for classification.
"""
import os
import sys
import random
import argparse
import copy

import torch
import torch.nn as nn
import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.append(os.getcwd())

from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
from uer.macro_moe.encoder import MacroMoEEncoder


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        if args.encoder == "macro_moe":
            self.encoder = MacroMoEEncoder(args)
        elif args.encoder == "backbone_only":
            args_copy = copy.deepcopy(args)
            args_copy.encoder = "transformer"
            args_copy.is_moe = False
            print("Loading BackboneOnly classifier: Transformer backbone without router/adapters/experts")
            self.encoder = str2encoder["transformer"](args_copy)
        else:
            self.encoder = str2encoder[args.encoder](args)

        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.macro_load_balance = getattr(args, "macro_load_balance", 0.1)
        self.macro_route_loss_weight = getattr(args, "macro_route_loss_weight", 0.3)
        self.macro_use_route_label = getattr(args, "macro_use_route_label", False)

        self.class_weights = None
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)
        self.use_center_loss = False
        self.center_loss_weight = 0.0
        self.centers = nn.Parameter(torch.randn(self.labels_num, args.hidden_size) * 0.02)

        # Few-shot prototype head. It is disabled by default and only enabled
        # explicitly in few-shot runs to avoid affecting the full-shot pipeline.
        self.use_prototype_head = False
        self.prototype_alpha = 0.5
        self.prototype_metric = "l2"
        self.prototype_bank = None

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights

    def set_prototype_head(self, enabled=False, alpha=0.5, metric="l2"):
        self.use_prototype_head = enabled
        self.prototype_alpha = alpha
        self.prototype_metric = metric

    def set_prototypes(self, prototypes=None):
        self.prototype_bank = prototypes

    def set_center_loss(self, enabled=False, weight=0.0):
        self.use_center_loss = enabled
        self.center_loss_weight = weight

    def encode_features(self, src, seg):
        emb = self.embedding(src, seg)

        if isinstance(self.encoder, MacroMoEEncoder):
            output, gate_loss, expert_indices, router_logits, router_probs = self.encoder(emb, seg)
        else:
            output = self.encoder(emb, seg)
            gate_loss = 0.0
            expert_indices = None
            router_logits = None

        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]

        features = torch.tanh(self.output_layer_1(output))
        return features, gate_loss, expert_indices, router_logits

    def compute_proto_logits(self, features):
        if self.prototype_bank is None:
            return None

        prototypes = self.prototype_bank.to(features.device)
        if self.prototype_metric == "cosine":
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
            prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=-1)
            return torch.matmul(features, prototypes.transpose(0, 1))

        diff = features.unsqueeze(1) - prototypes.unsqueeze(0)
        return -torch.sum(diff * diff, dim=-1)

    def forward(self, src, tgt, seg, soft_tgt=None, route_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        features, gate_loss, expert_indices, router_logits = self.encode_features(src, seg)
        logits = self.output_layer_2(features)

        if self.use_prototype_head and self.prototype_bank is not None:
            proto_logits = self.compute_proto_logits(features)
            if proto_logits is not None:
                logits = self.prototype_alpha * logits + (1.0 - self.prototype_alpha) * proto_logits

        if tgt is None:
            return None, logits, expert_indices

        loss_fct = nn.NLLLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )

        if self.soft_targets and soft_tgt is not None:
            loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                   (1 - self.soft_alpha) * loss_fct(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
        else:
            loss = loss_fct(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))

        if self.use_center_loss and self.center_loss_weight > 0.0:
            batch_centers = self.centers.index_select(0, tgt.view(-1))
            center_loss = torch.mean(torch.sum((features - batch_centers) ** 2, dim=-1))
            loss = loss + self.center_loss_weight * center_loss

        if self.macro_use_route_label and route_tgt is not None and router_logits is not None:
            route_tgt = route_tgt.to(router_logits.device)
            route_loss = nn.CrossEntropyLoss()(router_logits, route_tgt.view(-1))
            loss = loss + self.macro_route_loss_weight * route_loss

        if isinstance(self.encoder, MacroMoEEncoder) and isinstance(gate_loss, torch.Tensor):
            loss = loss + self.macro_load_balance * gate_loss

        return loss, logits, expert_indices


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def build_class_weights(dataset, labels_num, power=0.5):
    label_counts = torch.zeros(labels_num, dtype=torch.float)
    for sample in dataset:
        label_counts[sample[1]] += 1.0

    safe_counts = torch.clamp(label_counts, min=1.0)
    class_weights = (safe_counts.sum() / (labels_num * safe_counts)).pow(power)
    class_weights = class_weights / class_weights.mean()
    return class_weights


def load_checkpoint(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        print("Initialize with pretrained model.")
        model.load_state_dict(
            load_checkpoint(
                args.pretrained_model_path,
                map_location={"cuda:1": "cuda:0", "cuda:2": "cuda:0", "cuda:3": "cuda:0"},
            ),
            strict=False,
        )
    else:
        print("Initialize with normal distribution.")
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay_rate": 0.0,
        },
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False
        )
    else:
        optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False, relative_step=False
        )

    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None, route_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        route_tgt_batch = route_tgt[i * batch_size: (i + 1) * batch_size] if route_tgt is not None else None
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size: (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch, route_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None, route_tgt_batch

    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size:]
        seg_batch = seg[instances_num // batch_size * batch_size:, :]
        route_tgt_batch = route_tgt[instances_num // batch_size * batch_size:] if route_tgt is not None else None
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size:, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch, route_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None, route_tgt_batch


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue

            line = line.rstrip("\n").split("\t")
            tgt = int(line[columns["label"]])
            route_tgt = None
            if getattr(args, "macro_use_route_label", False) and "route_label" in columns:
                route_tgt = int(line[columns["route_label"]])

            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]

            if "text_b" not in columns:
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else:
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN]
                )
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)

            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt, route_tgt))
            else:
                dataset.append((src, tgt, seg, route_tgt))
    return dataset


def compute_support_prototypes(args, dataset):
    if not getattr(args, "fewshot_use_prototype_head", False):
        return None
    if len(dataset) == 0:
        return None

    model = unwrap_model(args.model)
    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])
    labels = torch.LongTensor([sample[1] for sample in dataset])

    was_training = model.training
    model.eval()
    features_list = []
    with torch.no_grad():
        for start in range(0, src.size(0), args.batch_size):
            src_batch = src[start: start + args.batch_size].to(args.device)
            seg_batch = seg[start: start + args.batch_size].to(args.device)
            features, _, _, _ = model.encode_features(src_batch, seg_batch)
            features_list.append(features.detach().cpu())
    if was_training:
        model.train()

    all_features = torch.cat(features_list, dim=0)
    global_center = all_features.mean(dim=0)
    prototypes = []
    for label_id in range(args.labels_num):
        label_mask = labels == label_id
        if label_mask.any():
            prototypes.append(all_features[label_mask].mean(dim=0))
        else:
            prototypes.append(global_center.clone())
    return torch.stack(prototypes, dim=0)


def update_model_prototypes(args, support_dataset):
    if not getattr(args, "fewshot_use_prototype_head", False):
        return
    prototypes = compute_support_prototypes(args, support_dataset)
    unwrap_model(args.model).set_prototypes(prototypes)


def sample_fewshot_episode(dataset, labels_num, support_shots, rng):
    label_to_examples = {label_id: [] for label_id in range(labels_num)}
    for sample in dataset:
        label_to_examples[sample[1]].append(sample)

    support_set, query_set = [], []
    for label_id in range(labels_num):
        examples = label_to_examples[label_id]
        if len(examples) == 0:
            continue

        shuffled = examples[:]
        rng.shuffle(shuffled)
        support_count = min(support_shots, max(1, len(shuffled) - 1)) if len(shuffled) > 1 else 1
        support_examples = shuffled[:support_count]
        query_examples = shuffled[support_count:]
        if len(query_examples) == 0:
            query_examples = support_examples

        support_set.extend(support_examples)
        query_set.extend(query_examples)

    rng.shuffle(support_set)
    rng.shuffle(query_set)
    return support_set, query_set


def train_fewshot_episode(args, model, optimizer, scheduler, support_set, query_set):
    if len(query_set) == 0:
        return 0.0

    update_model_prototypes(args, support_set)

    src = torch.LongTensor([example[0] for example in query_set])
    tgt = torch.LongTensor([example[1] for example in query_set])
    seg = torch.LongTensor([example[2] for example in query_set])

    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[3] for example in query_set])
        if args.macro_use_route_label:
            route_tgt = torch.LongTensor([example[4] if example[4] is not None else -1 for example in query_set])
        else:
            route_tgt = None
    else:
        soft_tgt = None
        if args.macro_use_route_label:
            route_tgt = torch.LongTensor([example[3] if example[3] is not None else -1 for example in query_set])
        else:
            route_tgt = None

    total_loss = 0.0
    batch_count = 0
    for src_batch, tgt_batch, seg_batch, soft_tgt_batch, route_tgt_batch in batch_loader(
        args.batch_size, src, tgt, seg, soft_tgt, route_tgt
    ):
        loss = train_model(
            args,
            model,
            optimizer,
            scheduler,
            src_batch,
            tgt_batch,
            seg_batch,
            soft_tgt_batch,
            route_tgt_batch,
        )
        total_loss += loss.item()
        batch_count += 1
    return total_loss / max(batch_count, 1)


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None, route_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)
    if route_tgt_batch is not None:
        route_tgt_batch = route_tgt_batch.to(args.device)

    loss, _, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch, route_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, print_confusion_matrix=False, support_dataset=None):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])
    batch_size = args.batch_size

    if getattr(args, "fewshot_use_prototype_head", False):
        update_model_prototypes(args, support_dataset if support_dataset is not None else dataset)

    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    y_true, y_pred = [], []
    all_expert_indices = []
    args.model.eval()

    route_tgt = None
    if getattr(args, "macro_use_route_label", False):
        route_index = 4 if args.soft_targets else 3
        route_tgt = torch.LongTensor([
            sample[route_index] if sample[route_index] is not None else -1 for sample in dataset
        ])

    for src_batch, tgt_batch, seg_batch, _, route_tgt_batch in batch_loader(
        batch_size, src, tgt, seg, route_tgt=route_tgt
    ):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        if route_tgt_batch is not None:
            route_tgt_batch = route_tgt_batch.to(args.device)

        with torch.no_grad():
            _, logits, expert_indices = args.model(src_batch, tgt_batch, seg_batch, route_tgt=route_tgt_batch)

        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
            y_true.append(gold[j].cpu())
            y_pred.append(pred[j].cpu())

            if expert_indices is not None:
                if expert_indices.dim() == 1:
                    all_expert_indices.append(expert_indices[j].cpu().item())
                else:
                    all_expert_indices.append(expert_indices[j].cpu().tolist())

        correct += torch.sum(pred == gold).item()

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        print("Report precision, recall, and f1:")
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

        if len(all_expert_indices) > 0:
            import json

            inferred_top_k = 1
            if any(isinstance(expert, list) for expert in all_expert_indices):
                inferred_top_k = max(len(expert) if isinstance(expert, list) else 1 for expert in all_expert_indices)
            routing_data = {
                "true_labels": [int(y) for y in y_true],
                "expert_indices": all_expert_indices,
                "top_k": inferred_top_k,
                "rank_expert_indices": [expert if isinstance(expert, list) else [expert] for expert in all_expert_indices],
            }
            with open("routing_analysis.json", "w") as f:
                json.dump(routing_data, f)
            print("routing_analysis.json saved!")

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    print(
        "Macro precision: {:.4f}, Micro precision: {:.4f}, Weighted precision: {:.4f}".format(
            precision_score(y_true, y_pred, average="macro", zero_division=0),
            precision_score(y_true, y_pred, average="micro", zero_division=0),
            precision_score(y_true, y_pred, average="weighted", zero_division=0),
        )
    )
    print(
        "Macro recall: {:.4f}, Micro recall: {:.4f}, Weighted recall: {:.4f}".format(
            recall_score(y_true, y_pred, average="macro", zero_division=0),
            recall_score(y_true, y_pred, average="micro", zero_division=0),
            recall_score(y_true, y_pred, average="weighted", zero_division=0),
        )
    )
    print(
        "Macro f1: {:.4f}, Micro f1: {:.4f}, Weighted f1: {:.4f}".format(
            f1_score(y_true, y_pred, average="macro", zero_division=0),
            f1_score(y_true, y_pred, average="micro", zero_division=0),
            f1_score(y_true, y_pred, average="weighted", zero_division=0),
        )
    )

    return f1_score(y_true, y_pred, average="macro", zero_division=0), confusion


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first", help="Pooling type.")
    parser.add_argument("--earlystop", type=int, default=5, help="early stop rounds.")
    parser.add_argument(
        "--tokenizer",
        choices=["bert", "char", "space"],
        default="bert",
        help="Specify the tokenizer.",
    )
    parser.add_argument("--soft_targets", action="store_true", help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5, help="Weight of the soft targets loss.")

    # Few-shot prototype head. Keep it off for pretraining/full-shot and enable
    # it only in few-shot runs when needed.
    parser.add_argument("--fewshot_use_prototype_head", action="store_true", help="Enable prototype head in few-shot runs.")
    parser.add_argument("--fewshot_proto_alpha", type=float, default=0.5, help="Fusion weight for classifier logits.")
    parser.add_argument("--fewshot_proto_metric", choices=["l2", "cosine"], default="l2", help="Prototype similarity metric.")
    parser.add_argument("--fewshot_episodic_train", action="store_true", help="Use episodic support/query training for few-shot.")
    parser.add_argument("--fewshot_support_shots", type=int, default=5, help="Number of support samples per class in each episode.")
    parser.add_argument("--fewshot_episodes_per_epoch", type=int, default=10, help="Number of episodes sampled in each epoch.")
    parser.add_argument("--fewshot_use_center_loss", action="store_true", help="Enable center loss for known-class low-resource training.")
    parser.add_argument("--fewshot_center_loss_weight", type=float, default=0.05, help="Weight for center loss.")

    # MOE Model Options
    parser.add_argument("--is_moe", action="store_true", help="adopt moe layer.")
    parser.add_argument("--vocab_size", type=int, required=False, help="Number of vocab.")
    parser.add_argument("--moebert_expert_dim", type=int, required=False, default=3072, help="Dim of expert,default is ffn.")
    parser.add_argument("--moebert_expert_num", type=int, required=False, help="Number of expert.")
    parser.add_argument(
        "--moebert_route_method",
        choices=["gate-token", "gate-sentence", "hash-random", "hash-balance", "proto"],
        default="hash-random",
        help="moebert route method.",
    )
    parser.add_argument("--moebert_route_hash_list", default=None, type=str, help="Path of moebert hash list file.")
    parser.add_argument("--moebert_load_balance", type=float, default=0.1, help="gate loss weight.")

    args = parser.parse_args()
    args = load_hyperparam(args)
    set_seed(args.seed)

    if args.fewshot_episodic_train and not args.fewshot_use_prototype_head:
        print("Enable prototype head automatically for episodic few-shot training.")
        args.fewshot_use_prototype_head = True

    if args.train_path is None:
        args.labels_num = 197
    else:
        args.labels_num = count_labels_num(args.train_path)

    args.tokenizer = str2tokenizer[args.tokenizer](args)
    model = Classifier(args)

    if args.encoder == "macro_moe":
        if hasattr(args, "few_shot_stage") and args.few_shot_stage:
            print("Enable Few-shot Adaptation Mode: Freezing Backbone, Training Adapters.")
            model.encoder.set_adaptation_mode(True)
        else:
            print("Enable Full Fine-tuning Mode: Unfreezing Backbone.")
            model.encoder.set_adaptation_mode(False)

    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    model.set_prototype_head(
        enabled=args.fewshot_use_prototype_head,
        alpha=args.fewshot_proto_alpha,
        metric=args.fewshot_proto_metric,
    )
    model.set_center_loss(
        enabled=args.fewshot_use_center_loss,
        weight=args.fewshot_center_loss_weight,
    )

    if args.train_path is None:
        args.model = model
        args.labels_num = 197
        print("No train data, only evaluate..")
        result = evaluate(args, read_dataset(args, args.dev_path))
        return

    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    class_weights = build_class_weights(trainset, args.labels_num)
    model.set_class_weights(class_weights.to(args.device))
    print("Class weights:", ["{:.3f}".format(weight) for weight in class_weights.tolist()])

    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[3] for example in trainset])
        if args.macro_use_route_label:
            route_tgt = torch.LongTensor([example[4] if example[4] is not None else -1 for example in trainset])
        else:
            route_tgt = None
    else:
        soft_tgt = None
        if args.macro_use_route_label:
            route_tgt = torch.LongTensor([example[3] if example[3] is not None else -1 for example in trainset])
        else:
            route_tgt = None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, best_result = 0.0, 0.0
    best_result_round = 0

    devset = read_dataset(args, args.dev_path)
    testset = read_dataset(args, args.test_path) if args.test_path is not None else None

    episode_rng = random.Random(args.seed)

    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()

        if args.fewshot_episodic_train:
            epoch_loss = 0.0
            for episode_id in range(args.fewshot_episodes_per_epoch):
                support_set, query_set = sample_fewshot_episode(
                    trainset,
                    args.labels_num,
                    args.fewshot_support_shots,
                    episode_rng,
                )
                episode_loss = train_fewshot_episode(args, model, optimizer, scheduler, support_set, query_set)
                epoch_loss += episode_loss

            avg_episode_loss = epoch_loss / max(args.fewshot_episodes_per_epoch, 1)
            print("Epoch id: {}, Episodic avg loss: {:.3f}".format(epoch, avg_episode_loss))
        else:
            if args.fewshot_use_prototype_head:
                update_model_prototypes(args, trainset)

            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch, route_tgt_batch) in enumerate(
                batch_loader(batch_size, src, tgt, seg, soft_tgt, route_tgt)
            ):
                loss = train_model(
                    args,
                    model,
                    optimizer,
                    scheduler,
                    src_batch,
                    tgt_batch,
                    seg_batch,
                    soft_tgt_batch,
                    route_tgt_batch,
                )
                total_loss += loss.item()
                if (i + 1) % args.report_steps == 0:
                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                    total_loss = 0.0

        result = evaluate(args, devset, support_dataset=trainset)
        if result[0] > best_result:
            best_result = result[0]
            best_result_round = epoch
            save_model(model, args.output_model_path)
        elif epoch - best_result_round >= args.earlystop:
            print("early stopping...")
            break

    if args.test_path is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(load_checkpoint(args.output_model_path))
        else:
            model.load_state_dict(load_checkpoint(args.output_model_path))
        args.model = model
        evaluate(args, testset, True, support_dataset=trainset)


if __name__ == "__main__":
    main()
