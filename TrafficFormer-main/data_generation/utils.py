import os, random, json, csv  # 导入操作系统、随机数、JSON和CSV模块
import ipaddress, pickle  # 导入IP地址处理和pickle序列化模块
import platform

# generate random ipv4 address
def random_ipv4():  # 生成随机IPv4地址的函数
    IPV4_MAX = ipaddress.IPv4Address._ALL_ONES  # 获取IPv4地址的最大值
    ip_int = random.randint(0, IPV4_MAX)  # 生成随机整数
    ip_str = ipaddress.IPv4Address._string_from_ip_int(ip_int)  # 将整数转换为IPv4地址字符串
    return ip_str  # 返回IPv4地址字符串


# generate random ipv6 address
def random_ipv6():  # 生成随机IPv6地址的函数
    IPV6_MAX = ipaddress.IPv6Address._ALL_ONES  # 获取IPv6地址的最大值
    ip_int = random.randint(0, IPV6_MAX)  # 生成随机整数
    ip_str = ipaddress.IPv6Address._string_from_ip_int(ip_int)  # 将整数转换为IPv6地址字符串
    return ip_str  # 返回IPv6地址字符串


def random_field(bits):  # 生成随机字段的函数
    field_max = 2 ** bits - 1  # 计算字段的最大值（2的bits次方减1）
    field_int = random.randint(0, field_max)  # 生成随机整数
    return field_int  # 返回随机整数


def convert_pcapng_2_pcap(pcapng_path, pcapng_file, output_path):  # 将pcapng转换为pcap的函数
    pcap_file = output_path + pcapng_file.replace('pcapng', 'pcap')  # 构建输出文件路径（替换扩展名）
    cmd = "editcap -F pcap %s %s"  # 定义转换命令模板
    command = cmd % (pcapng_path + pcapng_file, pcap_file)  # 构建完整命令
    os.system(command)  # 执行系统命令
    return 0  # 返回成功代码


# 这段代码是一个 调用 SplitCap 工具的自动化包装函数，用于批量、分类地 将原始 pcap 文件分割成多个会话级（或流级）的小 pcap 文件，方便后续的流量特征提取或模型训练。
'''def split_cap(pcap_split_path, pcap_file_path, pcap_name, pcap_label='', split_way='bidirection'):  # 分割pcap文件的函数
    # pcap_split_path + "splitcap/" + pcap_label + "/" + pcap_name is output
    # pcap_file_path+pcap_name is input
    if not os.path.exists(pcap_split_path + "/splitcap"):  # 如果分割目录不存在
        os.mkdir(pcap_split_path + "/splitcap")  # 创建分割目录
    if pcap_label != '':  # 如果提供了pcap标签
        if not os.path.exists(pcap_split_path + "splitcap/" + pcap_label):  # 如果标签目录不存在
            os.mkdir(pcap_split_path + "splitcap/" + pcap_label)  # 创建标签目录
        # if not os.path.exists(pcap_split_path + "splitcap/" + pcap_label + "/" + pcap_name):  # 注释掉的代码：检查子目录
        #     os.mkdir(pcap_split_path + "splitcap/" + pcap_label + "/" + pcap_name)  # 注释掉的代码：创建子目录
        output_path = pcap_split_path + "splitcap/" + pcap_label  # + "/" + pcap_name  # 构建输出路径
    else:  # 如果没有提供pcap标签
        if not os.path.exists(pcap_split_path + "splitcap/" + pcap_name):  # 如果名称目录不存在
            os.mkdir(pcap_split_path + "splitcap/" + pcap_name)  # 创建名称目录
        output_path = pcap_split_path + "splitcap/" + pcap_name  # 构建输出路径
    split_way = "session" if split_way == 'bidirection' else "flow"  # 根据分割方式设置参数
    print(pcap_file_path + pcap_name, output_path)  # 打印输入和输出路径
    cmd = f"mono ./SplitCap.exe -r {pcap_file_path + pcap_name} -s {split_way} -o {output_path}"  # 构建分割命令
    # print(cmd)  # 注释掉的调试信息
    os.system(cmd)  # 执行分割命令
    return output_path  # 返回输出路径
'''


def split_cap(pcap_split_path, pcap_file_path, pcap_name, pcap_label='', split_way='bidirection'):
    import platform
    import os

    # 规范化路径，确保使用正确的路径分隔符
    pcap_split_path = os.path.normpath(pcap_split_path)
    pcap_file_path = os.path.normpath(pcap_file_path)

    # 确保splitcap目录存在
    splitcap_dir = os.path.join(pcap_split_path, "splitcap")
    if not os.path.exists(splitcap_dir):
        os.makedirs(splitcap_dir)

    # 修复：简化输出路径结构，避免过深的嵌套
    if pcap_label != '':
        # 如果有标签，创建标签子目录
        output_dir = os.path.join(splitcap_dir, pcap_label)
    else:
        # 如果没有标签，直接使用splitcap目录
        # 修复：不要为每个文件创建子目录，避免路径过长
        output_dir = splitcap_dir

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 构建完整的输入文件路径
    input_file = os.path.join(pcap_file_path, pcap_name)

    # 修复：检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return output_dir

    split_way = "session" if split_way == 'bidirection' else "flow"

    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")

    # 修复：使用更安全的命令构建方式
    if platform.system() == "Windows":
        # Windows下直接执行exe文件
        # 修复：使用正确的路径引用方式
        splitcap_exe = "SplitCap.exe"
        if not os.path.exists(splitcap_exe):
            # 如果当前目录没有，尝试在项目根目录查找
            splitcap_exe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SplitCap.exe")

        cmd = f'"{splitcap_exe}" -r "{input_file}" -s {split_way} -o "{output_dir}"'
    else:
        # 获取当前 utils.py 所在的绝对路径 (即 data_generation 文件夹)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 拼接出 SplitCap.exe 的绝对路径
        splitcap_exe = os.path.join(current_dir, "SplitCap.exe")

        # 使用绝对路径来执行 mono
        cmd = f'mono "{splitcap_exe}" -r "{input_file}" -s {split_way} -o "{output_dir}" -p 500'

    print(f"执行命令: {cmd}")

    try:
        # 修复：使用subprocess代替os.system，获得更好的错误处理
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        print(f"命令执行结果: {result.returncode}")
        if result.stdout:
            print(f"命令输出: {result.stdout}")
        if result.stderr:
            print(f"命令错误: {result.stderr}")

        # 检查输出目录是否生成了文件
        output_files = [f for f in os.listdir(output_dir) if f.endswith('.pcap')]
        print(f"生成的分割文件数量: {len(output_files)}")

    except Exception as e:
        print(f"执行命令时出错: {e}")
        return output_dir

    return output_dir

def cut(obj, sec):  # 分割对象的函数
    result = [obj[i:i + sec] for i in range(0, len(obj), sec)]  # 将对象按指定长度分割
    try:  # 尝试执行
        remanent_count = len(result[0]) % 4  # 计算第一个元素的长度除以4的余数
    except Exception as e:  # 捕获异常
        remanent_count = 0  # 设置余数为0
        print("cut datagram error!")  # 打印错误信息
    if remanent_count == 0:  # 如果余数为0
        pass  # 什么都不做
    else:  # 如果余数不为0
        result = [obj[i:i + sec + remanent_count] for i in range(0, len(obj), sec + remanent_count)]  # 重新分割对象（考虑余数）
    return result  # 返回分割结果


# 将网络数据包的十六进制字符串转换为相邻字符对的序列
def bigram_generation(packet_datagram, token_len=64, flag=True):  # 生成bigram特征的函数
    result = ''  # 初始化结果字符串
    generated_datagram = cut(packet_datagram, 1)  # 将数据报文按1个字符分割
    token_count = 0  # 初始化token计数
    for sub_string_index in range(len(generated_datagram)):  # 遍历分割后的数据报文
        if sub_string_index != (len(generated_datagram) - 1):  # 如果不是最后一个元素
            token_count += 1  # token计数加1
            if token_count > token_len:  # 如果超过token长度限制
                break  # 跳出循环
            else:  # 如果没有超过长度限制
                merge_word_bigram = generated_datagram[sub_string_index] + generated_datagram[
                    sub_string_index + 1]  # 合并当前字符和下一个字符形成bigram
        else:  # 如果是最后一个元素
            break  # 跳出循环
        result += merge_word_bigram  # 将bigram添加到结果字符串
        result += ' '  # 添加空格分隔符

    return result  # 返回结果字符串


# 将网络数据包的十六进制字符串转换为固定长度的字符块
def gram_generation(packet_datagram):  # 生成gram特征的函数
    result = ''  # 初始化结果字符串
    generated_datagram = cut(packet_datagram, 2)  # 将数据报文按2个字符分割
    for sub_string_index in range(len(generated_datagram)):  # 遍历分割后的数据报文
        merge_word_bigram = generated_datagram[sub_string_index]  # 获取当前gram
        result += merge_word_bigram  # 将gram添加到结果字符串
        result += ' '  # 添加空格分隔符

    return result  # 返回结果字符串


def size_format(size):  # 格式化文件大小的函数
    # 'KB'
    file_size = '%.3f' % float(size / 1000)  # 将字节转换为KB并格式化
    return file_size  # 返回格式化后的文件大小


def read_data_from_json(json_data, features):  # 从JSON数据读取数据的函数
    X, Y = [], []  # 初始化特征和标签列表
    for feature_index in range(len(features)):  # 遍历特征索引
        x = []  # 初始化特征列表
        for label in json_data.keys():  # 遍历JSON数据的键（标签）
            x_label = []  # 初始化标签特征列表
            for sample_index in json_data[label][features[feature_index]].keys():  # 遍历样本索引
                x_label.append(json_data[label][features[feature_index]][sample_index])  # 添加特征值
            x.append(x_label)  # 添加标签特征列表
            y = [label] * len(x_label)  # 创建标签列表
            Y.append(y)  # 添加标签列表
        X.append(x)  # 添加特征列表
    return X, Y  # 返回特征和标签


def obtain_data(features, dataset_save_path, json_data=None):  # 获取数据的函数
    if json_data:  # 如果提供了JSON数据
        X, Y = read_data_from_json(json_data, features)  # 从JSON数据读取数据
    else:  # 如果没有提供JSON数据
        print("read dataset from json file.")  # 打印读取数据集信息
        with open(dataset_save_path + "/dataset.json", "r") as f:  # 打开数据集JSON文件
            dataset = json.load(f)  # 加载JSON数据
        X, Y = read_data_from_json(dataset, features)  # 从数据集读取数据

    for index in range(len(X)):  # 遍历特征列表
        if len(X[index]) != len(Y):  # 如果特征数量和标签数量不匹配
            print("data and labels are not properly associated.")  # 打印错误信息
            print("x:%s\ty:%s" % (len(X[index]), len(Y)))  # 打印特征和标签数量
            return -1  # 返回错误代码
    return X, Y  # 返回特征和标签


def write_dataset_tsv(data, label, file_dir, type):  # 写入数据集TSV文件的函数
    dataset_file = [["label", "text_a"]]  # 初始化数据集文件（表头）
    for index in range(len(label)):  # 遍历标签
        dataset_file.append([label[index], data[index]])  # 添加数据行
    with open(file_dir + type + "_dataset.tsv", 'w', newline='') as f:  # 打开TSV文件
        tsv_w = csv.writer(f, delimiter='\t')  # 创建CSV写入器（制表符分隔）
        tsv_w.writerows(dataset_file)  # 写入所有行
    return 0  # 返回成功代码


def write_dataset_tsv_twoc(data1, data2, label, file_dir, type):  # 写入双列数据集TSV文件的函数
    dataset_file = [["label", "text_a", "text_b"]]  # 初始化数据集文件（表头，双文本）
    for index in range(len(label)):  # 遍历标签
        dataset_file.append([label[index], data1[index], data2[index]])  # 添加数据行
    with open(file_dir + type + "_dataset.tsv", 'w', newline='') as f:  # 打开TSV文件
        tsv_w = csv.writer(f, delimiter='\t')  # 创建CSV写入器（制表符分隔）
        tsv_w.writerows(dataset_file)  # 写入所有行
    return 0  # 返回成功代码


def unlabel_data(label_data):  # 去除标签数据的函数
    nolabel_data = ""  # 初始化无标签数据字符串
    with open(label_data, newline='') as f:  # 打开标签数据文件
        data = csv.reader(f, delimiter='\t')  # 创建CSV读取器（制表符分隔）
        for row in data:  # 遍历数据行
            nolabel_data += row[1] + '\n'  # 添加文本数据（第二列）
    nolabel_file = label_data.replace("test_dataset", "nolabel_test_dataset")  # 构建无标签文件路径
    # nolabel_file = label_data.replace("train_dataset", "nolabel_train_dataset")  # 注释掉的代码：构建训练集无标签文件路径
    with open(nolabel_file, 'w', newline='') as f:  # 打开无标签文件
        f.write(nolabel_data)  # 写入无标签数据
    return 0  # 返回成功代码


# print(gram_generation("a86bad1f9bcdc49a025996f808004500003c0cc540004006b3b90a2a00d31f0d"))  # 注释掉的测试代码

def get_instance_number(file):  # 获取实例数量的函数
    count = 0  # 初始化计数
    with open(file, "rb") as f:  # 打开文件（二进制读取）
        try:  # 尝试执行
            while True:  # 无限循环
                intsance = pickle.load(f)  # 加载pickle对象
                count += 1  # 计数加1
                if count % 1000000 == 0:  # 每1000000个实例打印一次
                    print(count)  # 打印计数
        except EOFError:  # 捕获文件结束异常
            print(count)  # 打印最终计数


def typicalsamling(group, typicalNDict):  # 典型抽样函数
    name = group.name  # 获取组名
    n = typicalNDict[name]  # 获取该组的抽样数量
    return group.sample(n=n)  # 返回抽样结果
