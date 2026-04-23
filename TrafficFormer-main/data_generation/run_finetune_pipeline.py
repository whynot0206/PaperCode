import os
import sys

# 1. 确保能导入同级目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. 导入 finetuning_data_gen 中的核心函数
try:
    from finetuning_data_gen import convert_splitcap, generation_multiP, dataset_extract, enhance_based_tsv
except ImportError as e:
    print("错误：无法导入 finetuning_data_gen。请确保该文件在当前目录下。")
    print(f"详情: {e}")
    sys.exit(1)


def main():
    print("=== 开始执行微调数据生成流程 (包含数据增强) ===\n")

    # ==========================================
    # 配置区域
    # ==========================================
    ROOT_DIR = "/home/xuke/why_code/trafficformer"

    # 原始数据路径
    RAW_PCAP_DIR = os.path.join(ROOT_DIR, "moe_dataset_finetuning/ISCX-Tor")
    SPLIT_OUTPUT_DIR = os.path.join(ROOT_DIR, "moe_dataset_finetuning/ISCX-Tor_split")
    DATASET_OUTPUT_DIR = os.path.join(ROOT_DIR, "ISCX-Tor_dataset")

    FEATURES = ['datagram', "length", "time", "direction", "message_type"]

    # 数据增强配置
    ENABLE_AUGMENTATION = True  # 是否开启数据增强
    AUGMENT_FACTOR = 5  # 增强倍数 (例如 5 表示数据量变为原来的 5 倍)

    # ==========================================
    # Step 1: 分割 PCAP (Convert & Split)
    # ==========================================

    print(f"--- [Step 1] 正在分割 PCAP 文件 ---")
    print(f"源路径: {RAW_PCAP_DIR}")
    print(f"输出路径: {SPLIT_OUTPUT_DIR}")

    # 确保输出目录存在
    if not os.path.exists(SPLIT_OUTPUT_DIR):
        os.makedirs(SPLIT_OUTPUT_DIR)

    # 调用函数: pcapng_path, pcap_path, pcap_split_path, is_pcap_label
    # 注意：这里源路径和转换路径设为一样，简化操作
    convert_splitcap(
        pcapng_path=RAW_PCAP_DIR,
        pcap_path=RAW_PCAP_DIR,
        pcap_split_path=SPLIT_OUTPUT_DIR,
        is_pcap_label=True  # 关键：按文件夹名作为标签
    )

    print("Step 1 完成。\n")

    # ==========================================
    # Step 2: 提取特征生成 dataset.json
    # ==========================================
    print(f"--- [Step 2] 正在提取特征生成 dataset.json ---")

    # 确定 splitcap 的子路径
    splitcap_dir = os.path.join(SPLIT_OUTPUT_DIR, "splitcap/")

    # 动态计算类别数量，以生成 samples 列表
    # (原代码需要传入 samples 列表来限制每类样本数)
    if not os.path.exists(splitcap_dir):
        print(f"错误：找不到分割后的目录 {splitcap_dir}，请检查 Step 1 是否成功。")
        return

    # 获取类别目录列表
    categories = [d for d in os.listdir(splitcap_dir) if os.path.isdir(os.path.join(splitcap_dir, d))]
    num_categories = len(categories)
    print(f"检测到 {num_categories} 个类别: {categories}")

    if num_categories == 0:
        print("错误：未检测到任何类别文件夹，请检查数据源。")
        return

    # 设置每个类别的最大样本数 (设大一点以包含所有数据)
    MAX_SAMPLES_PER_CLASS = 10000
    samples_list = [MAX_SAMPLES_PER_CLASS] * num_categories

    # 确保输出目录存在
    if not os.path.exists(DATASET_OUTPUT_DIR):
        os.makedirs(DATASET_OUTPUT_DIR)

    # 确保 temp 目录存在 (这是之前报错的根源之一)
    if not os.path.exists("./temp/"):
        os.makedirs("./temp/")

    # 调用函数
    generation_multiP(
        pcap_path=splitcap_dir,
        samples=samples_list,
        dataset_save_path=DATASET_OUTPUT_DIR,
        start_index=28  # 跳过IP头
    )

    print("Step 2 完成。\n")

    # ==========================================
    # Step 3: 生成 TSV 数据集 (Extract)
    # ==========================================
    print(f"--- [Step 3] 正在生成 TSV 数据集 ---")

    # 调用函数

    dataset_extract(
        dataset_save_path=DATASET_OUTPUT_DIR,
        features=FEATURES
    )
    print("Step 3 完成。\n")

    # ==========================================
    # Step 3.5: 数据增强 (RIFA)
    # ==========================================
    train_file_path = os.path.join(DATASET_OUTPUT_DIR, "dataset/")
    train_filename = "train_dataset.tsv"

    enhanced_prefix = f"train_enhance{AUGMENT_FACTOR}"

    # [关键修复]：文件名必须包含 "_dataset.tsv" 后缀
    # 因为 finetuning_data_gen.py 调用的 utils.write_dataset_tsv 会强制添加此后缀
    enhanced_real_filename = f"{enhanced_prefix}_dataset.tsv"

    final_train_file = os.path.join(train_file_path, enhanced_real_filename)

    if ENABLE_AUGMENTATION:
        print(f"--- [Step 3.5] 正在进行数据增强 (因子: {AUGMENT_FACTOR}) ---")

        if not os.path.exists(os.path.join(train_file_path, train_filename)):
            print(f"[错误] 找不到原始训练集 {train_filename}，无法增强。")
            return

        # 如果文件不存在，则调用增强函数生成
        if not os.path.exists(final_train_file):
            enhance_based_tsv(
                path=train_file_path,
                filename=train_filename,
                new_file_prefix=enhanced_prefix,
                enhance_factor=AUGMENT_FACTOR
            )
            print(f"数据增强完成！新训练集生成于: {final_train_file}")
        else:
            print(f"检测到增强文件已存在，跳过生成: {final_train_file}")
    else:
        print("--- [Step 3.5] 数据增强已跳过 ---")
        final_train_file = os.path.join(train_file_path, train_filename)

    # ==========================================
    # Step 4: 提示微调命令
    # ==========================================
    print("\n=== 准备就绪 ===")
    if os.path.exists(final_train_file):
        print("\n接下来请在终端运行微调命令 (使用增强后的数据):")
        print("-" * 60)
        print(f"cd {ROOT_DIR}")
        print(f"CUDA_VISIBLE_DEVICES=0 python3 fine-tuning/run_classifier.py \\")
        print(f"    --vocab_path models/encryptd_vocab.txt \\")
        print(f"    --train_path {final_train_file} \\")  # <--- 注意这里用了增强后的文件
        print(f"    --dev_path {DATASET_OUTPUT_DIR}/dataset/valid_dataset.tsv \\")
        print(f"    --test_path {DATASET_OUTPUT_DIR}/dataset/test_dataset.tsv \\")
        print(f"    --pretrained_model_path models/pretrain_model_bert.bin-90000 \\")
        print(f"    --output_model_path models/finetuned_model.bin \\")
        # 增强后的数据量变大了，建议 Epoch 适当减少，或者 Batch Size 增大
        print(f"    --epochs_num 10 --batch_size 32 --seq_length 320 --learning_rate 3e-5")
        print("-" * 60)
    else:
        print(f"错误：训练文件 {final_train_file} 未找到。")


if __name__ == "__main__":
    main()

'''
python3 fine-tuning/run_classifier.py \
     --vocab_path models/encryptd_vocab.txt \
     --train_path /home/xuke/why_node/TrafficFormer_node/ISCX-VPN_dataset/dataset/train_dataset.tsv \
     --dev_path /home/xuke/why_node/TrafficFormer_node/ISCX-VPN_dataset/dataset/valid_dataset.tsv \
     --test_path /home/xuke/why_node/TrafficFormer_node/ISCX-VPN_dataset/dataset/test_dataset.tsv \
     --pretrained_model_path models/pretrain_model_bert.bin-90000 \
     --output_model_path models/finetuned_model.bin \
     --epochs_num 4 --batch_size 32 --seq_length 320 --learning_rate 6e-5

python3 fine-tuning/run_classifier.py \
     --vocab_path models/encryptd_vocab.txt \
     --train_path /home/xuke/why_node/TrafficFormer_node/ISCX-VPN_dataset/dataset/train_dataset.tsv \
     --dev_path /home/xuke/why_node/TrafficFormer_node/ISCX-VPN_dataset/dataset/valid_dataset.tsv \
     --test_path /home/xuke/why_node/TrafficFormer_node/ISCX-VPN_dataset/dataset/test_dataset.tsv \
     --pretrained_model_path models/pretrain_model_bert.bin-90000 \
     --output_model_path models/finetuned_model.bin \
     --epochs_num 20 --batch_size 32 --seq_length 320 --learning_rate 6e-5

python3 fine-tuning/run_classifier.py \
     --vocab_path models/encryptd_vocab.txt \
     --train_path /home/xuke/why_node/TrafficFormer_node/ISCX-VPN_dataset/dataset/train_dataset.tsv \
     --dev_path /home/xuke/why_node/TrafficFormer_node/ISCX-VPN_dataset/dataset/valid_dataset.tsv \
     --test_path /home/xuke/why_node/TrafficFormer_node/ISCX-VPN_dataset/dataset/test_dataset.tsv \
     --pretrained_model_path models/pretrain_model_bert.bin-90000 \
     --output_model_path models/finetuned_model.bin \
     --epochs_num 20 --batch_size 32 --seq_length 320 --learning_rate 1e-4
     
     
python3 fine-tuning/run_classifier.py \
     --vocab_path models/encryptd_vocab.txt \
     --train_path /home/xuke/why_node/TrafficFormer_node/ISCX-VPN_dataset/dataset/train_dataset.tsv \
     --dev_path /home/xuke/why_node/TrafficFormer_node/ISCX-VPN_dataset/dataset/valid_dataset.tsv \
     --test_path /home/xuke/why_node/TrafficFormer_node/ISCX-VPN_dataset/dataset/test_dataset.tsv \
     --pretrained_model_path models/pretrain_model_bert.bin-90000 \
     --output_model_path models/finetuned_model.bin \
     --epochs_num 40 --batch_size 64 --seq_length 320 --learning_rate 1e-4
     
     
python3 fine-tuning/run_classifier.py \
    --vocab_path models/encryptd_vocab.txt \
    --train_path ISCX-VPN_dataset/dataset/train_enhance5_dataset.tsv \
    --dev_path ISCX-VPN_dataset/dataset/valid_dataset.tsv \
    --test_path ISCX-VPN_dataset/dataset/test_dataset.tsv \
    --pretrained_model_path models/pretrain_model_bert.bin-90000 \
    --output_model_path models/finetuned_model.bin \
    --epochs_num 20 \
    --earlystop 15 \
    --batch_size 32 \
    --seq_length 320 \
    --learning_rate 2e-5
'''
'''
--- [Step 2] 正在提取特征生成 dataset.json ---
检测到 12 个类别: ['audio', 'chat', 'file', 'mail', 'streaming', 'voip', 'vpn-audio', 'vpn-chat', 'vpn-file', 'vpn-mail', 'vpn-streaming', 'vpn-voip']
label number:  12
audio 0
chat 1
file 2
mail 3
streaming 4
voip 5
vpn-audio 6
vpn-chat 7
vpn-file 8
vpn-mail 9
vpn-streaming 10
vpn-voip 11

'''
'''
调整数据集，只包含VPN数据集
检测到 6 个类别: ['vpn-audio', 'vpn-chat', 'vpn-file', 'vpn-mail', 'vpn-streaming', 'vpn-voip']
label number:  6
vpn-audio 0
vpn-chat 1
vpn-file 2
vpn-mail 3
vpn-streaming 4
vpn-voip 5


--- [Step 3] 正在生成 TSV 数据集 ---
read dataset from json file.
category flow
0       368
1       146
2       321
3       138
4       166
5       65
all     1204
label number of train: 6, val: 6, test: 6.
finish generating pre-train's datagram dataset.
Please check in /home/xuke/why_code/trafficformer/ISCX-VPN_dataset/dataset
Step 3 完成。

'''
'''

--- [Step 3] 正在生成 TSV 数据集 ---
read dataset from json file.
category flow
0       491
1       139
2       433
3       307
4       228
5       306
6       484
7       460
8       500
9       476
10      477
11      492
12      429
13      402
14      56
15      496
16      484
17      495
18      498
19      171
20      16
21      495
22      485
23      448
24      498
25      495
26      326
27      155
28      469
29      491
30      494
31      151
32      497
33      475
34      168
35      495
36      258
37      495
38      476
39      495
40      493
41      257
42      498
43      134
44      474
45      218
46      475
47      496
48      399
49      138
50      401
51      497
52      171
53      491
54      497
55      216
56      488
57      463
58      498
59      485
60      473
61      389
62      163
63      480
64      494
65      358
66      496
67      418
68      488
69      490
70      392
71      386
72      204
73      368
74      170
75      191
76      491
77      143
78      486
79      500
80      497
81      498
82      407
83      443
84      153
85      176
86      490
87      491
88      489
89      495
90      160
91      276
92      104
93      260
94      466
95      496
96      457
97      487
98      140
99      118
100     486
101     494
102     284
103     474
104     135
105     490
106     222
107     190
108     430
109     500
110     356
111     311
112     428
113     207
114     340
115     491
116     475
117     495
118     491
119     485
all     45962
label number of train: 120, val: 120, test: 120.
finish generating pre-train's datagram dataset.
Please check in /home/xuke/why_code/trafficformer/cstnet-tls 1.3_dataset/dataset
Step 3 完成。

'''