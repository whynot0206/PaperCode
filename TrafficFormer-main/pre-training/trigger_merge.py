import os
import sys

# 动态获取根目录
current_script_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(root_path)


def manual_merge_logic(dataset_path, workers_num):
    """
    完全复刻自 data.py 中的逻辑，但作为独立函数运行
    """
    abs_dataset_path = os.path.abspath(dataset_path)
    # 获取临时文件夹路径：data_generation/data/temp_datasets
    temp_dir = os.path.join(os.path.dirname(abs_dataset_path), "temp_datasets")

    print(f"开始合并临时文件到: {abs_dataset_path}")
    print(f"临时文件目录: {temp_dir}")

    # 按照你代码中的逻辑，先确保目标文件是干净的
    if os.path.exists(abs_dataset_path):
        os.remove(abs_dataset_path)

    for i in range(workers_num):
        temp_file_path = os.path.join(temp_dir, f"dataset-tmp-{i}.pt")

        if not os.path.exists(temp_file_path):
            print(f"警告: 找不到文件 {temp_file_path}，跳过...")
            continue

        print(f"正在合并第 {i}/{workers_num} 个文件...")

        # 按照 data.py 的原始 chunk 读取逻辑
        try:
            with open(temp_file_path, "rb") as reader:
                with open(abs_dataset_path, "ab") as dataset_writer:
                    while True:
                        chunk = reader.read(1024 * 1024)  # 1MB chunk
                        if not chunk:
                            break
                        dataset_writer.write(chunk)
            # 合并完一个删除一个，节省空间（如果你想保留，可以注释掉下面这行）
            os.remove(temp_file_path)
        except Exception as e:
            print(f"合并文件 {i} 时出错: {e}")

    print("--- 所有数据集合并完成！ ---")


def main():
    # 路径确保指向你的预定位置
    dataset_path = os.path.join(root_path, "data_generation/data/pretrain_dataset.pt")
    workers_num = 80

    print(f"--- 启动手动合并程序 ---")
    manual_merge_logic(dataset_path, workers_num)


if __name__ == "__main__":
    main()