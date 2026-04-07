import os, sys  # 导入操作系统和系统模块
import random  # 导入随机数模块
import shutil  # 导入文件操作模块
import binascii  # 导入二进制和ASCII转换模块
import scapy.all as scapy  # 导入网络数据包处理库
from functools import reduce  # 从functools导入reduce函数
from flowcontainer.extractor import extract  # 从flowcontainer导入流量特征提取器
from utils import *  # 导入utils模块中的所有函数
from tqdm import tqdm  # 导入进度条显示模块
import multiprocessing as mp  # 导入多进程处理模块
import traceback  # 导入异常追踪模块
from scapy.error import Scapy_Exception
import platform
def user_excepthook(tp, val, tb):  # 用户异常处理函数
    # print the exception to standard error
    traceback.print_exc()  # 打印异常信息到标准错误


# Semantic lossless enhancement
# 这个函数不是简单地随机替换数值，而是保持网络流量的相对关系和时序模式，只改变具体的地址、端口等标识信息。
# 保持流量特征不变： 时序模式：数据包之间的时间间隔不变 ，大小关系：序列号、确认号的相对增量不变 ，流量方向：客户端和服务器的角色不变 ，协议行为：TCP握手、数据传输模式不变
# 改变标识信息：IP地址：防止模型记忆特定IP,端口号：防止模型记忆特定服务端口,具体数值：序列号、确认号的具体数值
# 数据包层级的增强，它通过修改网络流量数据包中的标识符来产生增强数据，保持流量的其他特征和语义不变。
def enhancement(packets, is_addr=True, is_port=True, ):  # 语义无损增强函数
    # IP6 FLOWLABEL,src,dst
    # IP：IPID，src, dst
    # TCP: seq, ack, sport, dport,
    # UDP: sport, dport,
    if not hasattr(packets[0], 'type'):  # some datasets do not have ether header  # 检查是否有以太网头
        new_packets = scapy.PacketList()  # 创建新的数据包列表
        for i in range(len(packets)):  # 遍历数据包
            if packets[i].src == packets[0].src:  # ICMP  # 如果是ICMP协议（源地址相同）
                # create a ether header
                ether = scapy.Ether(src="00:00:00:00:00:00", dst="ff:ff:ff:ff:ff:ff", type=0x0800)  # 创建以太网头（客户端到服务器）
            else:  # 如果不是ICMP协议
                ether = scapy.Ether(src="ff:ff:ff:ff:ff:ff", dst="00:00:00:00:00:00", type=0x0800)  # 创建以太网头（服务器到客户端）
            packet = ether / packets[i]  # 组合以太网头和数据包
            new_packets.append(packet)  # 添加到新数据包列表
    packets = new_packets  # 更新数据包列表
    first_forward_packet = packets[0].copy()  # 复制第一个正向数据包
    for packet_index in range(len(packets)):  # 遍历数据包
        if packets[packet_index].src != first_forward_packet.src:  # 找到第一个反向数据包
            first_backward_packet = packets[packet_index].copy()  # 复制第一个反向数据包
            break  # 跳出循环
    if first_forward_packet.type == 0x0800:  # 如果是IPv4
        replace_src = random_ipv4()  # 生成随机源IP
        replace_dst = random_ipv4()  # 生成随机目标IP
        replace_src_id = random_field(16)  # 生成随机源ID
        replace_dst_id = random_field(16)  # 生成随机目标ID
        if first_forward_packet.payload.proto == 6:  # 如果是TCP
            replace_sport = random_field(16)  # 生成随机源端口
            replace_dport = random_field(16)  # 生成随机目标端口
            replace_src_seq = random_field(32)  # 生成随机源序列号
            replace_dst_seq = random_field(32)  # 生成随机目标序列号
        elif first_forward_packet.payload.proto == 17:  # 如果是UDP
            replace_sport = random_field(16)  # 生成随机源端口
            replace_dport = random_field(16)  # 生成随机目标端口

    elif first_forward_packet.type == 0x86dd:  # 如果是IPv6
        replace_src = random_ipv6()  # 生成随机源IPv6
        replace_dst = random_ipv6()  # 生成随机目标IPv6
        replace_src_flowlabel = random_field(20)  # 生成随机源流标签
        replace_dst_flowlabel = random_field(20)  # 生成随机目标流标签
        if first_forward_packet.payload.nh == 6:  # 如果是TCP
            replace_sport = random_field(16)  # 生成随机源端口
            replace_dport = random_field(16)  # 生成随机目标端口
            replace_src_seq = random_field(32)  # 生成随机源序列号
            replace_dst_seq = random_field(32)  # 生成随机目标序列号
        elif first_forward_packet.payload.nh == 17:  # 如果是UDP
            replace_sport = random_field(16)  # 生成随机源端口
            replace_dport = random_field(16)  # 生成随机目标端口

    for packet_index in range(len(packets)):  # 遍历数据包
        if packets[packet_index].src == first_forward_packet.src:  # forward  # 如果是正向数据包
            if is_addr:  # 如果需要增强地址
                packets[packet_index].payload.src = replace_src  # 设置源地址
                packets[packet_index].payload.dst = replace_dst  # 设置目标地址
            if packets[packet_index].type == 0x0800:  # 如果是IPv4
                if first_forward_packet.payload.id != 0:  # 如果原始IPID不为0
                    packets[packet_index].payload.id = replace_src_id + (
                            packets[packet_index].payload.id - first_forward_packet.payload.id)  # 计算新的IPID
                    packets[packet_index].payload.id %= 2 ** 16  # 取模防止溢出
                # print("forward: ",packets[packet_index].payload.id)  # 注释掉的调试信息
                if packets[packet_index].payload.proto == 6:  # 如果是TCP
                    if is_port:  # 如果需要增强端口
                        packets[packet_index].payload.payload.sport = replace_sport  # 设置源端口
                        packets[packet_index].payload.payload.dport = replace_dport  # 设置目标端口
                    packets[packet_index].payload.payload.seq = replace_src_seq + (packets[
                                                                                       packet_index].payload.payload.seq - first_forward_packet.payload.payload.seq)  # 计算新的序列号
                    if not ("S" in packets[packet_index].payload.payload.flags and packets[
                        packet_index].payload.payload.ack == 0):  # 如果不是SYN包
                        packets[packet_index].payload.payload.ack = replace_dst_seq + (packets[
                                                                                           packet_index].payload.payload.ack - first_backward_packet.payload.payload.seq)  # 计算新的确认号
                    packets[packet_index].payload.payload.seq %= 2 ** 32  # 取模防止溢出
                    packets[packet_index].payload.payload.ack %= 2 ** 32  # 取模防止溢出
                elif packets[packet_index].payload.proto == 17:  # 如果是UDP
                    if is_port:  # 如果需要增强端口
                        packets[packet_index].payload.payload.sport = replace_sport  # 设置源端口
                        packets[packet_index].payload.payload.dport = replace_dport  # 设置目标端口

            elif packets[packet_index].type == 0x86dd:  # 如果是IPv6
                packets[packet_index].payload.fl = replace_src_flowlabel + (
                        packets[packet_index].payload.fl - first_forward_packet.payload.fl)  # 计算新的流标签
                packets[packet_index].payload.fl %= 2 ** 20  # 取模防止溢出
                if packets[packet_index].payload.nh == 6:  # 如果是TCP
                    if is_port:  # 如果需要增强端口
                        packets[packet_index].payload.payload.sport = replace_sport  # 设置源端口
                        packets[packet_index].payload.payload.dport = replace_dport  # 设置目标端口
                    packets[packet_index].payload.payload.seq = replace_src_seq + (packets[
                                                                                       packet_index].payload.payload.seq - first_forward_packet.payload.payload.seq)  # 计算新的序列号
                    if not ("S" in packets[packet_index].payload.payload.flags and packets[
                        packet_index].payload.payload.ack == 0):  # 如果不是SYN包
                        packets[packet_index].payload.payload.ack = replace_dst_seq + (packets[
                                                                                           packet_index].payload.payload.ack - first_backward_packet.payload.payload.seq)  # 计算新的确认号
                    packets[packet_index].payload.payload.seq %= 2 ** 32  # 取模防止溢出
                    packets[packet_index].payload.payload.ack %= 2 ** 32  # 取模防止溢出
                elif packets[packet_index].payload.nh == 17:  # 如果是UDP
                    if is_port:  # 如果需要增强端口
                        packets[packet_index].payload.payload.sport = replace_sport  # 设置源端口
                        packets[packet_index].payload.payload.dport = replace_dport  # 设置目标端口
        else:  # backward  # 如果是反向数据包
            if is_addr:  # 如果需要增强地址
                packets[packet_index].payload.src = replace_dst  # 设置源地址（反向）
                packets[packet_index].payload.dst = replace_src  # 设置目标地址（反向）
            if packets[packet_index].type == 0x0800:  # 如果是IPv4
                packets[packet_index].payload.id = replace_dst_id + (
                        packets[packet_index].payload.id - first_backward_packet.payload.id)  # 计算新的IPID
                packets[packet_index].payload.id %= 2 ** 16  # 取模防止溢出
                # print("backward: ",replace_dst_id, packets[packet_index].payload.id)  # 注释掉的调试信息
                if packets[packet_index].payload.proto == 6:  # 如果是TCP
                    if is_port:  # 如果需要增强端口
                        packets[packet_index].payload.payload.sport = replace_dport  # 设置源端口（反向）
                        packets[packet_index].payload.payload.dport = replace_sport  # 设置目标端口（反向）
                    packets[packet_index].payload.payload.seq = replace_dst_seq + (packets[
                                                                                       packet_index].payload.payload.seq - first_backward_packet.payload.payload.seq)  # 计算新的序列号
                    if not ("S" in packets[packet_index].payload.payload.flags and packets[
                        packet_index].payload.payload.ack == 0):  # 如果不是SYN包
                        packets[packet_index].payload.payload.ack = replace_src_seq + (packets[
                                                                                           packet_index].payload.payload.ack - first_forward_packet.payload.payload.seq)  # 计算新的确认号
                    packets[packet_index].payload.payload.seq %= 2 ** 32  # 取模防止溢出
                    packets[packet_index].payload.payload.ack %= 2 ** 32  # 取模防止溢出
                elif packets[packet_index].payload.proto == 17:  # 如果是UDP
                    if is_port:  # 如果需要增强端口
                        packets[packet_index].payload.payload.sport = replace_dport  # 设置源端口（反向）
                        packets[packet_index].payload.payload.dport = replace_sport  # 设置目标端口（反向）
            elif packets[packet_index].type == 0x86dd:  # 如果是IPv6
                packets[packet_index].payload.fl = replace_dst_flowlabel + (
                        packets[packet_index].payload.fl - first_backward_packet.payload.fl)  # 计算新的流标签
                packets[packet_index].payload.fl %= 2 ** 20  # 取模防止溢出
                if packets[packet_index].payload.nh == 6:  # 如果是TCP
                    if is_port:  # 如果需要增强端口
                        packets[packet_index].payload.payload.sport = replace_sport  # 设置源端口（反向）
                        packets[packet_index].payload.payload.dport = replace_dport  # 设置目标端口（反向）
                    packets[packet_index].payload.payload.seq = replace_src_seq + (packets[
                                                                                       packet_index].payload.payload.seq - first_forward_packet.payload.payload.seq)  # 计算新的序列号
                    if not ("S" in packets[packet_index].payload.payload.flags and packets[
                        packet_index].payload.payload.ack == 0):  # 如果不是SYN包
                        packets[packet_index].payload.payload.ack = replace_src_seq + (packets[
                                                                                           packet_index].payload.payload.ack - first_forward_packet.payload.payload.seq)  # 计算新的确认号
                    packets[packet_index].payload.payload.seq %= 2 ** 32  # 取模防止溢出
                    packets[packet_index].payload.payload.ack %= 2 ** 32  # 取模防止溢出
                elif packets[packet_index].payload.nh == 17:  # 如果是UDP
                    if is_port:  # 如果需要增强端口
                        packets[packet_index].payload.payload.sport = replace_sport  # 设置源端口（反向）
                        packets[packet_index].payload.payload.dport = replace_dport  # 设置目标端口（反向）
    return packets  # 返回增强后的数据包


# 获取突发数据的函数，将每个流的连续包序列进一步划分成若干个“突发（burst）
'''def get_bursts(label_pcap, select_packet_len, corpora_path, start_index=0, enhance_factor=1,
               is_multi=False):
    if is_multi:  # 如果是多进程模式
        pid = os.getpid()  # 获取进程ID
    packets = scapy.rdpcap(label_pcap)  # 读取pcap文件
    No_ether = False  # 初始化无以太网头标志
    if not hasattr(packets[0], 'type'):  # no ether header  # 检查是否有以太网头
        # print("No ethernet...")  # 注释掉的调试信息
        No_ether = True  # 设置无以太网头标志
        # start_index -= 28  # 注释掉的代码：调整起始索引
        # return 0  # 注释掉的代码：返回0
    if (not No_ether and packets[0].type == 0x86dd) or (
            No_ether and packets[0].version == 6):  # not handle ipv6  # 检查是否是IPv6（不处理）
        return 0  # 返回0
    if len(packets) == 0:  # 如果没有数据包
        return 0  # 返回0

    packet_direction = []  # 初始化数据包方向列表
    feature_result = extract(label_pcap)  # 提取流量特征
    for key in feature_result.keys():  # 遍历特征键值
        value = feature_result[key]  # 获取特征值
        packet_direction = [x // abs(x) for x in value.ip_lengths]  # 计算数据包方向（1表示正向，-1表示反向）

    if len(packet_direction) == len(packets):  # 如果方向数量等于数据包数量
        burst_extra_info = ''  # 初始化突发额外信息

        if No_ether:  # 如果没有以太网头
            if packets[0].version == 4 and packets[0].proto == 6:  # 如果是IPv4 TCP
                burst_extra_info += '0'  # 添加协议标识'0'
            elif packets[0].version == 4 and packets[0].proto == 17:  # 如果是IPv4 UDP
                burst_extra_info += '1'  # 添加协议标识'1'
            else:  # 其他协议
                burst_extra_info += '2'  # 添加协议标识'2'
        else:  # 如果有以太网头
            if packets[0].type == 0x0800 and packets[0].payload.proto == 6:  # 如果是IPv4 TCP
                burst_extra_info += '0'  # 添加协议标识'0'
            elif packets[0].type == 0x0800 and packets[0].payload.proto == 17:  # 如果是IPv4 UDP
                burst_extra_info += '1'  # 添加协议标识'1'
            # elif packets[0].type == 0x86dd:  # 注释掉的代码：IPv6处理
            #     if packets[0].payload.nh == 6:  # 注释掉的代码：IPv6 TCP
            #         burst_extra_info += '2'  # 注释掉的代码：添加协议标识'2'
            #     elif packets[0].payload.nh == 17:  # 注释掉的代码：IPv6 UDP
            #         burst_extra_info += '3'  # 注释掉的代码：添加协议标识'3'
            else:  # 其他协议
                burst_extra_info += '2'  # 添加协议标识'2'
        burst_extra_info += '\n'  # 添加换行符

        burst_txt = ''  # 初始化突发文本
        for en in range(enhance_factor):  # 遍历增强因子
            if en > 0:  # 如果不是第一次迭代（需要增强）
                packets = enhancement(packets)  # 增强数据包

            packetss = []  # 初始化数据包列表（用于分批处理）
            packet_directionss = []  # 初始化方向列表（用于分批处理）
            new_packet_direction = []  # 初始化新方向列表
            new_packets = scapy.PacketList()  # 创建新数据包列表
            for packet_index in range(len(packets)):  # 遍历数据包
                new_packets.append(packets[packet_index])  # 添加数据包到新列表
                new_packet_direction.append(packet_direction[packet_index])  # 添加方向到新列表
                if (packet_index + 1) % 100 == 0:  # 每100个数据包一批
                    packetss.append(new_packets)  # 添加到数据包列表
                    packet_directionss.append(new_packet_direction)  # 添加到方向列表
                    new_packets = scapy.PacketList()  # 重置新数据包列表
                    new_packet_direction = []  # 重置新方向列表
            if len(new_packets) > 0:  # 如果有剩余数据包
                packetss.append(new_packets)  # 添加到数据包列表
                packet_directionss.append(new_packet_direction)  # 添加到方向列表

            for pp in range(len(packetss)):  # 遍历分批的数据包
                packets = packetss[pp]  # 获取当前批数据包
                packet_direction = packet_directionss[pp]  # 获取当前批方向
                burst_data_string = ''  # 初始化突发数据字符串
                for packet_index in range(len(packets)):  # 遍历当前批数据包
                    packet_data = packets[packet_index].copy()  # 复制数据包
                    data = (binascii.hexlify(bytes(packet_data)))  # 将数据包转换为十六进制字符串

                    if No_ether:  # add ether header  # 如果没有以太网头
                        packet_string = data.decode()  # 解码字节数据
                        if packet_direction[packet_index] == 1:  # 如果是正向数据包
                            packet_string = "c49a025996f8e46f13e2e3ae0800" + packet_string  # 添加以太网头（客户端到服务器）
                        else:  # 如果是反向数据包
                            packet_string = "e46f13e2e3aec49a025996f80800" + packet_string  # 添加以太网头（服务器到客户端）
                        packet_string = packet_string[start_index:start_index + 2 * select_packet_len]  # 截取指定长度的数据
                    else:  # 如果有以太网头
                        packet_string = data.decode()[start_index:start_index + 2 * select_packet_len]  # 直接截取指定长度的数据

                    if packet_index == 0:  # 如果是第一个数据包
                        packet_string = "||" + packet_string  # a new flow  # 添加流分隔符
                        burst_data_string += packet_string  # 添加到突发数据字符串
                    else:  # 如果不是第一个数据包
                        if packet_direction[packet_index] != packet_direction[packet_index - 1]:  # 如果方向改变

                            length = len(burst_data_string)  # 获取突发数据字符串长度
                            for string_txt in cut(burst_data_string, int(length / 2)):  # 将字符串分成两半
                                burst_txt += string_txt  # 添加到突发文本
                                # burst_txt += bigram_generation(string_txt, packet_len=len(string_txt))  # 注释掉的代码：生成bigram特征
                                burst_txt += '\n'  # 添加换行符
                            burst_txt += '\n'  # 添加空行（分隔符）

                            burst_data_string = ''  # 重置突发数据字符串

                        burst_data_string += packet_string  # 添加数据包字符串到突发数据字符串
                        if packet_index == len(packets) - 1:  # 如果是最后一个数据包

                            length = len(burst_data_string)  # 获取突发数据字符串长度
                            for string_txt in cut(burst_data_string, int(length / 2)):  # 将字符串分成两半
                                burst_txt += string_txt  # 添加到突发文本
                                # burst_txt += bigram_generation(string_txt, packet_len=len(string_txt))  # 注释掉的代码：生成bigram特征
                                burst_txt += '\n'  # 添加换行符
                            burst_txt += '\n'  # 添加空行（分隔符）
        if is_multi:  # 如果是多进程模式
            with open(corpora_path + "{}_biburst.txt".format(pid), 'a') as f:  # 打开进程特定文件
                f.write(burst_txt)  # 写入突发文本
        else:  # 如果是单进程模式
            with open(corpora_path, 'a') as f:  # 打开输出文件
                f.write(burst_txt)  # 写入突发文本

    return 0  # 返回成功代码'''


def get_bursts(label_pcap, select_packet_len, corpora_path, start_index=0, enhance_factor=1,
               is_multi=False):
    if is_multi:  # 如果是多进程模式
        pid = os.getpid()  # 获取进程ID

    # === 关键修正：添加 try-except 容错块来处理文件读取错误 ===
    try:
        packets = scapy.rdpcap(label_pcap)  # 读取pcap文件
    except Scapy_Exception as e:
        # 遇到无法读取的数据时（如空文件），打印警告并跳过此文件
        # 使用 sys.stderr 确保在多进程环境中警告信息能被打印出来
        print(f"WARNING: 进程 {pid if is_multi else 'Main'} 忽略损坏或空文件: {label_pcap}. 错误: {e}", file=sys.stderr)
        return 0  # 返回0，跳过此文件，防止进程崩溃
    # =========================================================

    No_ether = False  # 初始化无以太网头标志
    if not hasattr(packets[0], 'type'):  # no ether header  # 检查是否有以太网头
        # print("No ethernet...")  # 注释掉的调试信息
        No_ether = True  # 设置无以太网头标志
        # start_index -= 28  # 注释掉的代码：调整起始索引
        # return 0  # 注释掉的代码：返回0
    if (not No_ether and packets[0].type == 0x86dd) or (
            No_ether and packets[0].version == 6):  # not handle ipv6  # 检查是否是IPv6（不处理）
        return 0  # 返回0
    if len(packets) == 0:  # 如果没有数据包
        # 即使 Scapy 没有抛出异常，这里也检查是否为空列表
        if is_multi:
            print(f"WARNING: 进程 {pid} 忽略空文件 (0个数据包): {label_pcap}", file=sys.stderr)
        return 0  # 返回0

    packet_direction = []  # 初始化数据包方向列表
    # 注意：extract() 函数调用了 flowcontainer 库，如果该库在处理特定文件时也崩溃，需要额外的 try-except 包裹它
    try:
        feature_result = extract(label_pcap)  # 提取流量特征
        for key in feature_result.keys():  # 遍历特征键值
            value = feature_result[key]  # 获取特征值
            packet_direction = [x // abs(x) for x in value.ip_lengths]  # 计算数据包方向（1表示正向，-1表示反向）
    except Exception as e:
        print(f"ERROR: 进程 {pid if is_multi else 'Main'} 在提取特征时失败: {label_pcap}. 错误: {e}", file=sys.stderr)
        return 0  # 跳过特征提取失败的文件

    if len(packet_direction) == len(packets):  # 如果方向数量等于数据包数量
        burst_extra_info = ''  # 初始化突发额外信息

        if No_ether:  # 如果没有以太网头
            if packets[0].version == 4 and packets[0].proto == 6:  # 如果是IPv4 TCP
                burst_extra_info += '0'  # 添加协议标识'0'
            elif packets[0].version == 4 and packets[0].proto == 17:  # 如果是IPv4 UDP
                burst_extra_info += '1'  # 添加协议标识'1'
            else:  # 其他协议
                burst_extra_info += '2'  # 添加协议标识'2'
        else:  # 如果有以太网头
            if packets[0].type == 0x0800 and packets[0].payload.proto == 6:  # 如果是IPv4 TCP
                burst_extra_info += '0'  # 添加协议标识'0'
            elif packets[0].type == 0x0800 and packets[0].payload.proto == 17:  # 如果是IPv4 UDP
                burst_extra_info += '1'  # 添加协议标识'1'
            # elif packets[0].type == 0x86dd:  # 注释掉的代码：IPv6处理
            #     if packets[0].payload.nh == 6:  # 注释掉的代码：IPv6 TCP
            #         burst_extra_info += '2'  # 注释掉的代码：添加协议标识'2'
            #     elif packets[0].payload.nh == 17:  # 注释掉的代码：IPv6 UDP
            #         burst_extra_info += '3'  # 注释掉的代码：添加协议标识'3'
            else:  # 其他协议
                burst_extra_info += '2'  # 添加协议标识'2'
        burst_extra_info += '\n'  # 添加换行符

        burst_txt = ''  # 初始化突发文本
        for en in range(enhance_factor):  # 遍历增强因子
            if en > 0:  # 如果不是第一次迭代（需要增强）
                packets = enhancement(packets)  # 增强数据包

            packetss = []  # 初始化数据包列表（用于分批处理）
            packet_directionss = []  # 初始化方向列表（用于分批处理）
            new_packet_direction = []  # 初始化新方向列表
            new_packets = scapy.PacketList()  # 创建新数据包列表
            for packet_index in range(len(packets)):  # 遍历数据包
                new_packets.append(packets[packet_index])  # 添加数据包到新列表
                new_packet_direction.append(packet_direction[packet_index])  # 添加方向到新列表
                if (packet_index + 1) % 100 == 0:  # 每100个数据包一批
                    packetss.append(new_packets)  # 添加到数据包列表
                    packet_directionss.append(new_packet_direction)  # 添加到方向列表
                    new_packets = scapy.PacketList()  # 重置新数据包列表
                    new_packet_direction = []  # 重置新方向列表
            if len(new_packets) > 0:  # 如果有剩余数据包
                packetss.append(new_packets)  # 添加到数据包列表
                packet_directionss.append(new_packet_direction)  # 添加到方向列表

            for pp in range(len(packetss)):  # 遍历分批的数据包
                packets = packetss[pp]  # 获取当前批数据包
                packet_direction = packet_directionss[pp]  # 获取当前批方向
                burst_data_string = ''  # 初始化突发数据字符串
                for packet_index in range(len(packets)):  # 遍历当前批数据包
                    packet_data = packets[packet_index].copy()  # 复制数据包
                    data = (binascii.hexlify(bytes(packet_data)))  # 将数据包转换为十六进制字符串

                    if No_ether:  # add ether header  # 如果没有以太网头
                        packet_string = data.decode()  # 解码字节数据
                        if packet_direction[packet_index] == 1:  # 如果是正向数据包
                            packet_string = "c49a025996f8e46f13e2e3ae0800" + packet_string  # 添加以太网头（客户端到服务器）
                        else:  # 如果是反向数据包
                            packet_string = "e46f13e2e3aec49a025996f80800" + packet_string  # 添加以太网头（服务器到客户端）
                        packet_string = packet_string[start_index:start_index + 2 * select_packet_len]  # 截取指定长度的数据
                    else:  # 如果有以太网头
                        packet_string = data.decode()[start_index:start_index + 2 * select_packet_len]  # 直接截取指定长度的数据

                    if packet_index == 0:  # 如果是第一个数据包
                        packet_string = "||" + packet_string  # a new flow  # 添加流分隔符
                        burst_data_string += packet_string  # 添加到突发数据字符串
                    else:  # 如果不是第一个数据包
                        if packet_direction[packet_index] != packet_direction[packet_index - 1]:  # 如果方向改变

                            length = len(burst_data_string)  # 获取突发数据字符串长度
                            for string_txt in cut(burst_data_string, int(length / 2)):  # 将字符串分成两半
                                burst_txt += string_txt  # 添加到突发文本
                                # burst_txt += bigram_generation(string_txt, packet_len=len(string_txt))  # 注释掉的代码：生成bigram特征
                                burst_txt += '\n'  # 添加换行符
                            burst_txt += '\n'  # 添加空行（分隔符）

                            burst_data_string = ''  # 重置突发数据字符串

                        burst_data_string += packet_string  # 添加数据包字符串到突发数据字符串
                        if packet_index == len(packets) - 1:  # 如果是最后一个数据包

                            length = len(burst_data_string)  # 获取突发数据字符串长度
                            for string_txt in cut(burst_data_string, int(length / 2)):  # 将字符串分成两半
                                burst_txt += string_txt  # 添加到突发文本
                                # burst_txt += bigram_generation(string_txt, packet_len=len(string_txt))  # 注释掉的代码：生成bigram特征
                                burst_txt += '\n'  # 添加换行符
                            burst_txt += '\n'  # 添加空行（分隔符）
        if is_multi:  # 如果是多进程模式
            with open(corpora_path + "{}_biburst.txt".format(pid), 'a') as f:  # 打开进程特定文件
                f.write(burst_txt)  # 写入突发文本
        else:  # 如果是单进程模式
            with open(corpora_path, 'a') as f:  # 打开输出文件
                f.write(burst_txt)  # 写入突发文本

    return 0  # 返回成功代码

# 获取连续数据包的函数，从原始流量包中提取连续的、属于同一个流（flow）的数据包序列。
def _extract_packet_window(packet_data, packet_direction, start_index, select_packet_len, no_ether):
    data = binascii.hexlify(bytes(packet_data)).decode()
    if no_ether:
        if packet_direction == 1:
            data = "c49a025996f8e46f13e2e3ae0800" + data
        else:
            data = "e46f13e2e3aec49a025996f80800" + data
    end_index = start_index + 2 * select_packet_len
    return data[start_index:end_index]


def _overwrite_hex_field(packet_string, byte_offset, byte_length, fill_hex="00"):
    start = byte_offset * 2
    end = start + byte_length * 2
    if end > len(packet_string):
        return packet_string
    return packet_string[:start] + (fill_hex * byte_length) + packet_string[end:]


def sanitize_packet_window(packet_string, proto_type):
    """
    Keep the packet window shape unchanged while removing shortcut-prone identifiers.
    The offsets assume the packet window starts from the IPv4 header.
    """
    if len(packet_string) < 40:
        return packet_string

    sanitized = packet_string
    for byte_offset, byte_length in ((4, 2), (12, 4), (16, 4)):
        sanitized = _overwrite_hex_field(sanitized, byte_offset, byte_length)

    if proto_type == 6:
        for byte_offset, byte_length in ((20, 2), (22, 2), (24, 4), (28, 4)):
            sanitized = _overwrite_hex_field(sanitized, byte_offset, byte_length)
    elif proto_type == 17:
        for byte_offset, byte_length in ((20, 2), (22, 2)):
            sanitized = _overwrite_hex_field(sanitized, byte_offset, byte_length)

    return sanitized


def get_bursts_moe(label_pcap, select_packet_len, corpora_path, start_index=0, enhance_factor=1,
                   is_multi=False):
    if is_multi:
        pid = os.getpid()

    try:
        packets = scapy.rdpcap(label_pcap)
    except Scapy_Exception as e:
        print(f"WARNING: worker {pid if is_multi else 'Main'} failed to read pcap: {label_pcap}. error: {e}",
              file=sys.stderr)
        return 0

    if len(packets) == 0:
        if is_multi:
            print(f"WARNING: worker {pid} encountered empty pcap: {label_pcap}", file=sys.stderr)
        return 0

    no_ether = not hasattr(packets[0], 'type')
    if (not no_ether and packets[0].type == 0x86dd) or (no_ether and packets[0].version == 6):
        return 0

    packet_direction = []
    try:
        feature_result = extract(label_pcap)
        for key in feature_result.keys():
            value = feature_result[key]
            packet_direction = [x // abs(x) for x in value.ip_lengths]
    except Exception as e:
        print(f"ERROR: worker {pid if is_multi else 'Main'} failed to extract flow features: {label_pcap}. error: {e}",
              file=sys.stderr)
        return 0

    if len(packet_direction) != len(packets):
        return 0

    burst_txt = ''
    for en in range(enhance_factor):
        current_packets = packets if en == 0 else enhancement(packets)
        packetss = []
        packet_directionss = []
        new_packet_direction = []
        new_packets = scapy.PacketList()
        for packet_index in range(len(current_packets)):
            new_packets.append(current_packets[packet_index])
            new_packet_direction.append(packet_direction[packet_index])
            if (packet_index + 1) % 100 == 0:
                packetss.append(new_packets)
                packet_directionss.append(new_packet_direction)
                new_packets = scapy.PacketList()
                new_packet_direction = []
        if len(new_packets) > 0:
            packetss.append(new_packets)
            packet_directionss.append(new_packet_direction)

        for pp in range(len(packetss)):
            chunk_packets = packetss[pp]
            chunk_directions = packet_directionss[pp]
            burst_data_string = ''
            for packet_index in range(len(chunk_packets)):
                packet_data = chunk_packets[packet_index].copy()
                packet_string = _extract_packet_window(
                    packet_data,
                    chunk_directions[packet_index],
                    start_index,
                    select_packet_len,
                    no_ether,
                )

                if no_ether:
                    proto_type = packet_data.proto if hasattr(packet_data, "proto") else -1
                else:
                    proto_type = packet_data.payload.proto if packet_data.type == 0x0800 else -1
                packet_string = sanitize_packet_window(packet_string, proto_type)

                if packet_index == 0:
                    packet_string = "||" + packet_string
                    burst_data_string += packet_string
                else:
                    if chunk_directions[packet_index] != chunk_directions[packet_index - 1]:
                        length = len(burst_data_string)
                        for string_txt in cut(burst_data_string, int(length / 2)):
                            burst_txt += string_txt
                            burst_txt += '\n'
                        burst_txt += '\n'
                        burst_data_string = ''

                    burst_data_string += packet_string
                    if packet_index == len(chunk_packets) - 1:
                        length = len(burst_data_string)
                        for string_txt in cut(burst_data_string, int(length / 2)):
                            burst_txt += string_txt
                            burst_txt += '\n'
                        burst_txt += '\n'

    if is_multi:
        with open(corpora_path + "{}_biburst.txt".format(pid), 'a') as f:
            f.write(burst_txt)
    else:
        with open(corpora_path, 'a') as f:
            f.write(burst_txt)
    return 0


def get_consecutive_packets(label_pcap, select_packet_len, corpora_path, start_index=0):
    packets = scapy.rdpcap(label_pcap)  # 读取pcap文件

    if not hasattr(packets[0], 'type'):  # 检查是否有以太网头
        print("No ethernet...")  # 打印无以太网头信息
        return 0  # 返回0

    packet_direction = []  # 初始化数据包方向列表
    feature_result = extract(label_pcap)  # 提取流量特征
    for key in feature_result.keys():  # 遍历特征键值
        value = feature_result[key]  # 获取特征值
        packet_direction = [x // abs(x) for x in value.ip_lengths]  # 计算数据包方向

    if len(packet_direction) == len(packets):  # 如果方向数量等于数据包数量
        burst_txt = ''  # 初始化突发文本
        burst_direction = ''  # 初始化突发方向字符串
        for packet_index in range(len(packets)):  # 遍历数据包
            packet_data = packets[packet_index].copy()  # 复制数据包
            data = (binascii.hexlify(bytes(packet_data)))  # 将数据包转换为十六进制字符串

            packet_string = data.decode()[start_index:start_index + 2 * select_packet_len]  # 截取指定长度的数据

            if packet_index == 0:  # 如果是第一个数据包
                burst_txt += packet_string  # 添加到突发文本
                burst_txt += '\n'  # 添加换行符
            else:  # 如果不是第一个数据包
                burst_txt += packet_string  # 添加到突发文本
                burst_txt += '\n'  # 添加换行符
                burst_txt += '\n'  # 添加空行
                burst_txt += packet_string  # 再次添加数据包字符串（用于上下文）
                if packet_direction[packet_index] != packet_direction[packet_index - 1]:  # 如果方向改变
                    burst_direction += '0'  # 添加方向改变标记'0'
                else:  # 如果方向不变
                    burst_direction += '1'  # 添加方向不变标记'1'

        with open(corpora_path, 'a') as f:  # 打开输出文件
            f.write(burst_txt)  # 写入突发文本
        with open(corpora_path[:-4] + "_extra.txt", 'a') as f:  # 打开额外信息文件
            f.write(burst_direction)  # 写入突发方向信息
    return 0  # 返回成功代码


def merge(path):  # 合并文件的函数
    pid_set = set()  # 初始化进程ID集合
    for filename in os.listdir(path):  # 遍历目录中的文件
        pid_set.add(filename.split('_')[0])  # 提取进程ID并添加到集合
    with open(path[:-1] + "_biburst.txt", 'w') as fw1:  # 打开合并后的输出文件
        # with open(path[:-1]+"_extra.txt",'w') as fw2:  # 注释掉的代码：打开合并后的额外信息文件
        for key in pid_set:  # 遍历进程ID集合
            with open(path + key + "_biburst.txt", 'r') as fr:  # 打开进程特定文件
                while True:  # 无限循环
                    line = fr.readline()  # 读取一行
                    if not line:  # 如果到达文件末尾
                        break  # 跳出循环
                    fw1.write(line)  # 写入合并文件
            # with open(path + key+"_extra.txt",'r') as fr:  # 注释掉的代码：打开进程特定额外信息文件
            #     while True:  # 注释掉的代码：无限循环
            #         line = fr.readline()  # 注释掉的代码：读取一行
            #         if not line:  # 注释掉的代码：如果到达文件末尾
            #             break  # 注释掉的代码：跳出循环
            #         fw2.write(line)  # 注释掉的代码：写入合并文件


# 预训练数据集生成函数，将前面的突发数据，进一步转换成模型可读的语料（例如token序列、mask序列等）
def pretrain_dataset_generation(pcapng_path, pcap_output_path, output_split_path, select_packet_len, corpora_path,
                                start_index=0, enhance_factor=1, is_multi=True):
    # pcapng_path: the path of pcapng files (if the traffic is the pacp type, pcapng_path = pcap_output_path)
    # pcap_output_path: the path of pcap files
    # output_split_path: the path of splited pcap files
    # start_index: the index of start byte (the number of byte * 2, i.e., If start from IP header, start_index = 28)
    # select_packet_len: the bytes of used
    # enhance_factor: enhance factor, enhance_factor = 1 represents do not enhance the pretrain data
    # is_multi: use multi process or not, the number of process is set at line: pool = mp.Pool(100)

    if not os.listdir(pcap_output_path):  # 如果pcap输出路径为空
        print("Begin to convert pcapng to pcap.")  # 打印开始转换信息
        for _parent, _dirs, files in os.walk(pcapng_path):  # 遍历pcapng路径
            for file in files:  # 遍历文件
                if 'pcapng' in file:  # 如果是pcapng文件
                    # print(_parent + file)  # 注释掉的调试信息
                    convert_pcapng_2_pcap(_parent, file, pcap_output_path)  # 转换为pcap格式
                else:  # 如果不是pcapng文件
                    shutil.copy(_parent + "/" + file, pcap_output_path + file)  # 直接复制文件

    if not os.path.exists(output_split_path + "splitcap"):  # 如果分割目录不存在
        print("Begin to split pcap as session flows.")  # 打印开始分割信息
        for _p, _d, files in os.walk(pcap_output_path):  # 遍历pcap输出路径
            for file in files:  # 遍历文件
                split_cap(output_split_path, pcap_output_path, file)  # 分割pcap文件

    print("Begin to generate burst dataset.")
    if is_multi:
        all_files = []
        for _p, _d, files in os.walk(output_split_path + "splitcap"):
            for file in files:
                all_files.append(_p + "/" + file)
        pbar = tqdm(total=len(all_files))
        pbar.set_description('get bursts')
        update = lambda *args: pbar.update()
        if not os.path.exists(corpora_path):
            os.makedirs(corpora_path)
        pool = mp.Pool(processes=10)
        results = []
        for file in all_files:
            result = pool.apply_async(get_bursts,
                                      (file, select_packet_len, corpora_path, start_index, enhance_factor, True),
                                      callback=update)
            results.append(result)
        pool.close()
        pool.join()
        # 确保所有任务都已完成
        for result in results:
            result.get()  # 这会阻塞直到任务完成
        pbar.close()
        print("start merge files...")
        merge(corpora_path)

        if os.path.exists(corpora_path):
            shutil.rmtree(corpora_path)
    else:
        # 单进程模式保持不变
        for _p, _d, files in os.walk(output_split_path + "splitcap"):
            for file in tqdm(files):
                get_bursts(_p + "/" + file, select_packet_len=select_packet_len, corpora_path=corpora_path,
                           start_index=start_index, enhance_factor=enhance_factor)

    return 0
"""   print("Begin to generate burst dataset.")  # 打印开始生成突发数据集信息
    if is_multi:  # 如果是多进程模式
        all_files = []  # 初始化所有文件列表
        for _p, _d, files in os.walk(output_split_path + "splitcap"):  # 遍历分割目录
            for file in files:  # 遍历文件
                all_files.append(_p + "/" + file)  # 添加文件路径到列表
        pbar = tqdm(total=len(all_files))  # 初始化进度条
        pbar.set_description('get bursts')  # 设置进度条描述
        update = lambda *args: pbar.update()  # 定义更新函数

        if not os.path.exists(corpora_path):  # 如果语料库路径不存在
            os.makedirs(corpora_path)  # 创建语料库目录

        #pool = mp.Pool(100)  # 创建进程池（100个进程）
        pool = mp.Pool(mp.cpu_count())

        for file in all_files:  # 遍历所有文件
            pool.apply_async(get_bursts, (file, select_packet_len, corpora_path, start_index, enhance_factor, True),
                             callback=update)  # 异步调用获取突发函数
        pool.close()  # 关闭进程池
        pool.join()  # 等待所有进程完成
        print("start merge files...")  # 打印开始合并文件信息
        merge(corpora_path)  # 合并文件
        # os.system(f'rm -r {corpora_path}')  # 删除临时目录
        # 替换为更安全的Python原生方法:
        if os.path.exists(corpora_path):
            shutil.rmtree(corpora_path)
    else:  # 如果是单进程模式
        for _p, _d, files in os.walk(output_split_path + "splitcap"):  # 遍历分割目录
            for file in tqdm(files):  # 遍历文件（带进度条）
                get_bursts(_p + "/" + file, select_packet_len=select_packet_len, corpora_path=corpora_path,
                           start_index=start_index, enhance_factor=enhance_factor)  # 获取突发数据

    return 0  # 返回成功代码
"""

# 把突发序列转化为可直接用于语言模型的“语料（corpora）”文件
def corpora_to_bigram(corpora_path, corpora_bigram_path):  # 语料库转换为bigram的函数
    with open(corpora_bigram_path, 'w') as fw:  # 打开bigram输出文件
        with open(corpora_path, 'r') as fr:  # 打开语料库文件
            while True:  # 无限循环
                line = fr.readline()  # 读取一行
                if not line:  # 如果到达文件末尾
                    break  # 跳出循环
                if not line.strip():  # 如果是空行
                    fw.write(line)  # 直接写入空行
                else:  # 如果不是空行
                    newline = bigram_generation(line.strip(), token_len=len(line.strip()))  # 生成bigram特征
                    if newline[:2] == "||":  # 如果有流分隔符
                        newline = "||" + newline[5:]  # 调整分隔符位置
                    fw.write(newline + "\n")  # 写入bigram特征


# 把突发序列转化为可直接用于语言模型的“语料（corpora）”文件
def corpora_to_gram(corpora_path, corpora_gram_path):  # 语料库转换为gram的函数
    with open(corpora_gram_path, 'w') as fw:  # 打开gram输出文件
        with open(corpora_path, 'r') as fr:  # 打开语料库文件
            while True:  # 无限循环
                line = fr.readline()  # 读取一行
                if not line:  # 如果到达文件末尾
                    break  # 跳出循环
                if not line.strip():  # 如果是空行
                    fw.write(line)  # 直接写入空行
                else:  # 如果不是空行
                    if line[:2] == "||":  # 如果有流分隔符
                        newline = gram_generation(line.strip()[2:])  # 生成gram特征（去除分隔符）
                        newline = "||" + newline  # 重新添加分隔符
                    else:  # 如果没有分隔符
                        newline = gram_generation(line.strip())  # 生成gram特征
                    fw.write(newline + "\n")  # 写入gram特征


def read_flows(path):  # 读取流的函数
    print("process ", path)  # 打印处理文件信息
    file1 = []  # 初始化文件列表
    with open(path, 'r') as fr:  # 打开文件
        flow = []  # 初始化流列表
        while True:  # 无限循环
            line = fr.readline()  # 读取一行
            if not line:  # 如果到达文件末尾
                break  # 跳出循环
            if line[:2] == "||":  # 如果是流分隔符
                if len(flow) > 0:  # 如果当前流不为空
                    file1.append(flow)  # 添加到文件列表
                flow = []  # 重置流列表
                flow.append(line)  # 添加分隔符到流
            else:  # 如果不是分隔符
                flow.append(line)  # 添加行到流
        if len(flow) > 0:  # 如果最后一个流不为空
            file1.append(flow)  # 添加到文件列表
    return file1  # 返回文件列表


def merge_txts():  # 合并文本文件的函数
    corpora_path1 = "corpora1.txt"  # 语料库1路径
    corpora_path2 = "corpora2.txt"  # 语料库2路径
    corpora_path3 = "corpora3.txt"  # 语料库3路径
    corpora_path = "corpora.txt"  # 合并后语料库路径
    file1 = read_flows(corpora_path1)  # 读取语料库1
    for flow in file1:  # 遍历语料库1中的流
        if len(flow) > 100000:  # 如果流长度超过100000
            print(len(flow))  # 打印流长度

    file2 = read_flows(corpora_path2)  # 读取语料库2
    for flow in file2:  # 遍历语料库2中的流
        if len(flow) > 100000:  # 如果流长度超过100000
            print(len(flow))  # 打印流长度
    file3 = read_flows(corpora_path3)  # 读取语料库3
    for flow in file3:  # 遍历语料库3中的流
        if len(flow) > 100000:  # 如果流长度超过100000
            print(len(flow))  # 打印流长度
    files = file1 + file2 + file3  # 合并所有流
    random.shuffle(files)  # 随机打乱流
    return  # 返回
