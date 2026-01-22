import os
import random
import shutil
import binascii
import scapy.all as scapy
from functools import reduce
from flowcontainer.extractor import extract
from scapy.layers.tls.record import TLS
# from torch._numpy._dtypes_impl import _category # [修复1] 删除这行错误的导入！

from utils import *
import json, operator
from tqdm import tqdm
import pickle
import multiprocessing as mp
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.model_selection import train_test_split
import string
import argparse

scapy.load_layer("tls")
from scapy.layers.tls.handshake import TLSClientHello, TLSServerHello
from scapy.layers.tls.extensions import TLS_Ext_ServerName

def random_tcp_ts_option(packets):
    # 随机化TCP时间戳选项的函数
    src_ts = None  # 初始化源时间戳变量
    dst_ts = None  # 初始化目标时间戳变量
    random_src_ts = random_field(32)  # 生成32位随机源时间戳
    random_dst_ts = random_field(32)  # 生成32位随机目标时间戳
    src_port = None  # 初始化源端口变量
    for packet in packets:  # 遍历所有数据包
        if packet.haslayer(scapy.TCP):  # 检查数据包是否有TCP层
            tcp_options = [list(option) for option in packet['TCP'].options]  # 将TCP选项转换为列表形式
            for option in tcp_options:  # 遍历TCP选项
                if option[0] == 'Timestamp':  # 如果选项是时间戳类型
                    if src_port == None:  # 如果是第一个时间戳选项
                        src_port = packet['TCP'].sport  # 记录源端口
                        src_ts = option[1][0]  # 记录源时间戳
                        if option[1][1] != 0:  # 如果目标时间戳不为0
                            dst_ts = option[1][1]  # 记录目标时间戳
                    if packet['TCP'].sport == src_port:  # 如果数据包源端口等于记录的源端口
                        if option[1][1] != 0:  # 如果目标时间戳不为0
                            option[1] = (
                                random_src_ts + option[1][0] - src_ts,
                                random_dst_ts + option[1][1] - dst_ts)  # 计算新的时间戳值
                        else:  # 如果目标时间戳为0
                            option[1] = (random_src_ts + option[1][0] - src_ts, 0)  # 只更新源时间戳
                    else:  # 如果数据包源端口不等于记录的源端口（即反向流量）
                        if dst_ts == None:  # 如果目标时间戳未记录
                            dst_ts = option[1][0]  # 记录目标时间戳
                        if option[1][1] != 0:  # 如果目标时间戳不为0
                            option[1] = (
                                random_dst_ts + option[1][0] - dst_ts,
                                random_src_ts + option[1][1] - src_ts)  # 计算新的时间戳值
                        else:  # 如果目标时间戳为0
                            option[1] = (random_dst_ts + option[1][0] - dst_ts, 0)  # 只更新目标时间戳

            packet['TCP'].options = [tuple(option) for option in tcp_options]  # 将修改后的选项列表转换回元组形式
    return packets  # 返回修改后的数据包列表


def random_tls_sni():
    # 随机化TLS SNI（服务器名称指示）的函数
    # We're using few bytes and won't get to the sni field
    # server name(sni) is a variable-length field, modifying it triggers a series of length modifications, ranging from tls extension length, tls length, IP length, and even packet ack.
    scapy.load_layer("tls")  # 加载TLS层
    from scapy.layers.tls.handshake import TLSClientHello, TLSServerHello  # 导入TLS握手协议相关类
    from scapy.layers.tls.extensions import TLS_Ext_ServerName  # 导入TLS服务器名称扩展
    packets = scapy.rdpcap("1.pcap")  # 读取pcap文件中的数据包
    count = 0  # 初始化计数器
    for packet in packets:  # 遍历所有数据包
        # change server name
        print(count)  # 打印当前数据包计数
        count += 1  # 计数器加1
        if packet.haslayer(TLSClientHello):  # 检查数据包是否有TLS客户端问候层
            tls_client_hello = packet[TLSClientHello]  # 获取TLS客户端问候层
            tls_client_hello.show()  # 显示TLS客户端问候层详细信息
            print(packet['IP'].len, packet[TLS].len, tls_client_hello.msglen, tls_client_hello.extlen)  # 打印各种长度信息
            add = 0  # 初始化长度增加值
            if tls_client_hello.ext == None:  # 如果没有扩展字段
                continue  # 跳过当前数据包
            for ext in tls_client_hello.ext:  # 遍历所有扩展字段
                # Check for Server Name extension
                if isinstance(ext, TLS_Ext_ServerName):  # 如果是服务器名称扩展
                    for server_name in ext.servernames:  # 遍历所有服务器名称
                        random_length = random.randint(10, 25)  # random length  # 生成随机长度
                        # random_length = server_name.namelen #keep length  # 注释掉的代码：保持原长度
                        print(random_length)  # 打印随机长度
                        # ramdom server name
                        random_server_name = ''.join(
                            random.choices(string.ascii_lowercase + string.digits, k=random_length))  # 生成随机服务器名称
                        print(random_server_name)  # 打印随机服务器名称
                        server_name.servername = random_server_name.encode("UTF-8")  # 设置新的服务器名称
                        add += random_length - server_name.namelen  # 计算长度变化值
                        server_name.namelen = random_length  # 更新名称长度
                        server_name.nametype = 0  # 设置名称类型

                    print("add: ", add)  # 打印长度增加值
                    ext.len += add  # 更新扩展长度
                    ext.servernameslen += add  # 更新服务器名称列表长度

            tls_client_hello.extlen += add  # 更新TLS扩展长度
            tls_client_hello.msglen += add  # 更新TLS消息长度
            packet[TLS].len += add  # 更新TLS记录长度
            packet['IP'].len += add  # 更新IP包长度

        # change tls1.2 timestamps
        if packet.haslayer(TLSClientHello):  # 检查数据包是否有TLS客户端问候层
            packet[TLSClientHello].gmt_unix_time = random_field(32)  # 随机化TLS客户端时间戳
        if packet.haslayer(TLSServerHello):  # 检查数据包是否有TLS服务器问候层
            packet[TLSServerHello].gmt_unix_time = random_field(32)  # 随机化TLS服务器时间戳

    scapy.wrpcap("random_sni.pcap", packets)  # 将修改后的数据包写入新的pcap文件


def random_ip_port(packets: scapy.PacketList):
    # 随机化IP地址和端口的函数
    if len(packets) > 0:  # 如果数据包列表不为空
        if scapy.IP in packets[0]:  # 如果第一个数据包有IP层
            client_ip = packets[0][scapy.IP].src  # 获取客户端IP地址
            server_ip = packets[0][scapy.IP].dst  # 获取服务器IP地址
            r_client_ip = random_ipv4()  # 生成随机客户端IP地址
            r_server_ip = random_ipv4()  # 生成随机服务器IP地址
        elif scapy.IPv6 in packets[0]:  # 如果第一个数据包有IPv6层
            client_ip = packets[0][scapy.IPv6].src  # 获取客户端IPv6地址
            server_ip = packets[0][scapy.IPv6].dst  # 获取服务器IPv6地址
            r_client_ip = random_ipv6()  # 生成随机客户端IPv6地址
            r_server_ip = random_ipv6()  # 生成随机服务器IPv6地址

        else:  # 如果既不是IP也不是IPv6
            print("other L3 protocol")  # 打印提示信息
        r_client_port = random_field(16)  # 生成随机客户端端口
        r_server_port = random_field(16)  # 生成随机服务器端口
    else:  # 如果数据包列表为空
        print("No packet in the pcap file")  # 打印错误信息
        return -1  # 返回错误代码

    for packet in packets:  # 遍历所有数据包
        if scapy.IP in packet:  # 如果数据包有IP层
            is_client = packet[scapy.IP].src == client_ip  # 判断是否是客户端发出的数据包
            packet[scapy.IP].src = r_client_ip if is_client else r_server_ip  # 设置新的源IP地址
            packet[scapy.IP].dst = r_server_ip if is_client else r_client_ip  # 设置新的目标IP地址
            if scapy.TCP in packet:  # 如果数据包有TCP层
                packet[scapy.TCP].sport = r_client_port if is_client else r_server_port  # 设置新的源端口
                packet[scapy.TCP].dport = r_server_port if is_client else r_client_port  # 设置新的目标端口
            if scapy.UDP in packet:  # 如果数据包有UDP层
                packet[scapy.UDP].sport = r_client_port if is_client else r_server_port  # 设置新的源端口
                packet[scapy.UDP].dport = r_server_port if is_client else r_client_port  # 设置新的目标端口
        if scapy.IPv6 in packet:  # 如果数据包有IPv6层
            is_client = packet[scapy.IPv6].src == client_ip  # 判断是否是客户端发出的数据包
            packet[scapy.IPv6].src = r_client_ip if is_client else r_server_ip  # 设置新的源IPv6地址
            packet[scapy.IPv6].dst = r_server_ip if is_client else r_client_ip  # 设置新的目标IPv6地址
            if scapy.TCP in packet:  # 如果数据包有TCP层
                packet[scapy.TCP].sport = r_client_port if is_client else r_server_port  # 设置新的源端口
                packet[scapy.TCP].dport = r_server_port if is_client else r_client_port  # 设置新的目标端口
            if scapy.UDP in packet:  # 如果数据包有UDP层
                packet[scapy.UDP].sport = r_client_port if is_client else r_server_port  # 设置新的源端口
                packet[scapy.UDP].dport = r_server_port if is_client else r_client_port  # 设置新的目标端口

    return packets  # 返回修改后的数据包列表


def random_tls_randomtime(packets: scapy.PacketList):
    # 随机化TLS时间戳的函数
    for packet in packets:  # 遍历所有数据包
        if packet.haslayer(TLSClientHello):  # 检查数据包是否有TLS客户端问候层
            packet[TLSClientHello].gmt_unix_time = random_field(32)  # 随机化TLS客户端时间戳
        if packet.haslayer(TLSServerHello):  # 检查数据包是否有TLS服务器问候层
            packet[TLSServerHello].gmt_unix_time = random_field(32)  # 随机化TLS服务器时间戳
    return packets  # 返回修改后的数据包列表


# 这个函数把一个 .pcap 文件的原始网络包流，转化为一个“文本样本 + 数值特征”的组合 —— 它既保留了包级别的内容信息（作为 NLP 输入），
# 又保留了时间、长度、方向、TLS语义等流量特征，最终输出一条可用于模型训练的结构化样本。
def get_feature_flow(label_pcap, select_packet_len, packets_num, start_index=76, add_sep=True):
    # 从pcap文件中提取流量特征的函数
    feature_data = []  # 初始化特征数据列表
    packets = scapy.rdpcap(label_pcap)  # 读取pcap文件
    packet_count = 0  # 初始化数据包计数器
    flow_data_string = ''  # 初始化流量数据字符串

    No_ether = False  # 初始化以太网头标志
    if not hasattr(packets[0], 'type'):  # no ether header  # 检查第一个数据包是否有type属性（判断是否有以太网头）
        No_ether = True  # 设置无以太网头标志
    if (not No_ether and packets[0].type == 0x86dd) or (
            No_ether and packets[0].version == 6):  # do not handle IPV6  # 检查是否是IPv6（不处理）
        return -1  # 返回错误代码

    feature_result = extract(label_pcap, filter='tcp', extension=['tls.record.content_type', 'tls.record.opaque_type',
                                                                  'tls.handshake.type'])  # 提取TCP流量特征
    # 'tls.record.content_type'：TLS 记录的类型（例如，握手、应用数据、警报）。
    # 'tls.record.opaque_type'：这个字段在某些TLS版本或工具中可能用来指代记录类型。
    # 'tls.handshake.type'：TLS 握手消息的类型（例如，ClientHello, ServerHello, Certificate, Finished 等）。

    if len(feature_result) == 0:  # 如果没有提取到TCP特征
        feature_result = extract(label_pcap, filter='udp')  # 尝试提取UDP流量特征
        if len(feature_result) == 0:  # 如果也没有UDP特征
            return -1  # 返回错误代码
        extract_keys = list(feature_result.keys())[0]  # 获取特征结果键值
        if len(feature_result[label_pcap, extract_keys[1], extract_keys[2]].ip_lengths) < 3:  # 检查UDP流量的数据包数量
            print("preprocess udp flow %s but this flow has less than 3 packets." % label_pcap)  # 打印警告信息
            return -1  # 返回错误代码
    elif len(packets) < 3:  # 如果TCP流量数据包数量少于3
        print("preprocess tcp flow %s but this flow has less than 3 packets." % label_pcap)  # 打印警告信息
        return -1  # 返回错误代码
    try:  # 尝试执行以下代码
        if len(feature_result[label_pcap, 'tcp', '0'].ip_lengths) < 3:  # 检查TCP流量的数据包数量
            print("1: preprocess flow %s but this flow has less than 3 packets." % label_pcap)  # 打印警告信息
            return -1  # 返回错误代码
    except Exception as e:  # 捕获异常
        # print("*** this flow begins from 1 or other numbers than 0.")  # 注释掉的调试信息
        for key in feature_result.keys():  # 遍历所有特征键值
            if len(feature_result[key].ip_lengths) < 3:  # 检查每个流的数据包数量
                print("2: preprocess flow %s but this flow has less than 3 packets." % label_pcap)  # 打印警告信息
                return -1  # 返回错误代码

    if feature_result == {}:  # 如果特征结果为空
        return -1  # 返回错误代码

    packet_length = []  # 初始化数据包长度列表
    packet_time = []  # 初始化数据包时间列表
    packet_direction = []  # 初始化数据包方向列表
    packet_message_type = []  # 初始化数据包消息类型列表

    feature_result_lens = len(feature_result.keys())  # 获取特征结果数量
    for key in feature_result.keys():  # 遍历所有特征键值
        value = feature_result[key]  # 获取特征值
        packet_length.append(value.ip_lengths)  # 添加IP长度数据
        packet_time.append(value.ip_timestamps)  # 添加IP时间戳数据

        if len(packet_length) < feature_result_lens:  # 如果还没有处理完所有特征
            continue  # 继续下一个循环
        elif len(packet_length) == 1:  # 如果只有一个特征
            pass  # 什么都不做
        else:  # 如果有多个特征
            packet_length = [sum(packet_length, [])]  # 合并所有长度列表
            packet_time = [sum(packet_time, [])]  # 合并所有时间列表

        extension_dict = {}  # 初始化扩展字典

        for len_index in range(len(packet_length)):  # 遍历长度列表
            extension_list = [0] * (len(packet_length[len_index]))  # 初始化扩展列表

        extensions = value.extension  # 获取扩展信息

        if 'tls.record.content_type' in extensions.keys():  # 如果有TLS记录内容类型
            for record_content in extensions['tls.record.content_type']:  # 遍历TLS记录内容类型
                packet_index = record_content[1]  # 获取数据包索引
                ms_type = []  # 初始化消息类型列表

                if len(record_content[0]) > 2:  # 如果内容长度大于2
                    ms_type.extend(record_content[0].split(','))  # 分割字符串并扩展列表
                else:  # 如果内容长度小于等于2
                    ms_type.append(record_content[0])  # 直接添加到列表

                extension_dict[packet_index] = ms_type  # 将消息类型添加到扩展字典

            if 'tls.handshake.type' in extensions.keys():  # 如果有TLS握手类型
                for tls_handshake in extensions['tls.handshake.type']:  # 遍历TLS握手类型
                    packet_index = tls_handshake[1]  # 获取数据包索引
                    if packet_index not in extension_dict.keys():  # 如果数据包索引不在扩展字典中
                        continue  # 跳过当前循环
                    ms_type = []  # 初始化消息类型列表
                    if len(tls_handshake[0]) > 2:  # 如果内容长度大于2
                        ms_type.extend(tls_handshake[0].split(','))  # 分割字符串并扩展列表
                    else:  # 如果内容长度小于等于2
                        ms_type.append(tls_handshake[0])  # 直接添加到列表
                    source_length = len(extension_dict[packet_index])  # 获取源长度
                    for record_index in range(source_length):  # 遍历记录索引
                        if extension_dict[packet_index][record_index] == '22':  # 如果是握手记录
                            for handshake_type_index in range(len(ms_type)):  # 遍历握手类型
                                extension_dict[packet_index][record_index] = '22:' + ms_type[
                                    handshake_type_index]  # 组合握手类型
                                if handshake_type_index > 0:  # 如果不是第一个类型
                                    extension_dict[packet_index].insert(handshake_type_index,
                                                                        ('22:' + ms_type[
                                                                            handshake_type_index]))  # 插入新的握手类型
                            break  # 跳出循环
        if 'tls.record.opaque_type' in extensions.keys():  # 如果有TLS不透明类型
            for record_opaque in extensions['tls.record.opaque_type']:  # 遍历TLS不透明类型
                packet_index = record_opaque[1]  # 获取数据包索引
                ms_type = []  # 初始化消息类型列表
                if len(record_opaque[0]) > 2:  # 如果内容长度大于2
                    ms_type.extend(record_opaque[0].split(","))  # 分割字符串并扩展列表
                else:  # 如果内容长度小于等于2
                    ms_type.append(record_opaque[0])  # 直接添加到列表
                if packet_index not in extension_dict.keys():  # 如果数据包索引不在扩展字典中
                    extension_dict[packet_index] = ms_type  # 添加到扩展字典
                else:  # 如果数据包索引已在扩展字典中
                    extension_dict[packet_index].extend(ms_type)  # 扩展消息类型列表

        # extension_dict is {0: ['22:2', '20'], 16: ['20', '23'], 7: ['23'],...}

        is_source = 0  # 初始化源标志
        if is_source:  # one method record tls type  # 如果是源记录方法
            # {0: '22:2,20', 16: '20,23', 7: '23',...}
            extension_string_dict = {}  # 初始化扩展字符串字典
            for key in extension_dict.keys():  # 遍历扩展字典键值
                temp_string = ''  # 初始化临时字符串
                for status in extension_dict[key]:  # 遍历状态列表
                    temp_string += status + ','  # 拼接状态字符串
                temp_string = temp_string[:-1]  # 去除最后一个逗号
                extension_string_dict[key] = temp_string  # 添加到扩展字符串字典
            packet_message_type.append(extension_string_dict)  # 添加到数据包消息类型列表
        else:  # Another method record tls type  # 另一种记录TLS类型的方法
            # [64,...,23,...,43] [22*2+20,...,23,...,20+23]
            for key in extension_dict.keys():  # 遍历扩展字典键值
                if len(set(extension_dict[key])) == 1 and len(extension_dict[key]) > 1:  # 如果所有状态相同且数量大于1
                    try:  # 尝试执行
                        extension_list[key] += len(extension_dict[key])  # 增加扩展列表值
                    except Exception as e:  # 捕获异常
                        print(key)  # 打印键值
                else:  # 如果状态不同或数量为1
                    for status in extension_dict[key]:  # 遍历状态列表
                        if ':' in status:  # 如果状态包含冒号
                            extension_list[key] += reduce(operator.mul, [int(x) for x in status.split(':')],
                                                          1)  # 计算乘积并累加
                        else:  # 如果状态不包含冒号
                            if key <= len(packet_length[0]):  # 如果键值在合理范围内
                                extension_list[key] += int(status)  # 直接累加整数值
                            else:  # 如果键值超出范围
                                with open("error_while_writin_record", "a") as f:  # 打开错误记录文件
                                    f.write(label_pcap + '\n')  # 写入错误信息
                                continue  # 继续下一个循环
            packet_message_type.append(extension_list)  # 添加到数据包消息类型列表

    for length in packet_length[0]:  # 遍历数据包长度
        if length > 0:  # 如果长度大于0
            packet_direction.append(1)  # 添加正向方向标志
        else:  # 如果长度小于等于0
            packet_direction.append(-1)  # 添加反向方向标志

    packet_index = 0  # 初始化数据包索引

    packets = random_ip_port(packets)  # 随机化IP和端口
    packets = random_tcp_ts_option(packets)  # 随机化TCP时间戳选项
    packets = random_tls_randomtime(packets)  # 随机化TLS时间戳

    for packet in packets:  # 遍历所有数据包
        packet_data = packet.copy()  # 复制数据包
        data = (binascii.hexlify(bytes(packet_data)))  # 将数据包转换为十六进制字符串
        if No_ether:  # 如果没有以太网头
            packet_string = data.decode()  # 解码字节数据
            if packet_direction[packet_index] == 1:  # 如果是正向数据包
                packet_string = "c49a025996f8e46f13e2e3ae0800" + packet_string  # 添加以太网头（客户端到服务器）
            else:  # 如果是反向数据包
                packet_string = "e46f13e2e3aec49a025996f80800" + packet_string  # 添加以太网头（服务器到客户端）
            packet_string = packet_string[start_index:start_index + 2 * select_packet_len]  # 截取指定长度的数据
        else:  # 如果有以太网头
            packet_string = data.decode()[start_index:start_index + 2 * select_packet_len]  # 直接截取指定长度的数据

        if add_sep:  # 如果需要添加分隔符
            flow_data_string += "[SEP] "  # 添加分隔符
        flow_data_string += bigram_generation(packet_string.strip(), token_len=len(packet_string.strip()),
                                              flag=True)  # 生成bigram特征并添加到流量数据字符串
        # flow_data_string += gram_generation(packet_string.strip())  # 注释掉的代码：生成gram特征
        packet_count += 1  # 数据包计数器加1
        if packet_count == packets_num:  # 如果达到指定数据包数量
            break  # 跳出循环

    feature_data.append(flow_data_string)  # 添加流量数据字符串到特征数据
    feature_data.append(packet_length[0])  # 添加数据包长度到特征数据
    feature_data.append(packet_time[0])  # 添加数据包时间到特征数据
    feature_data.append(packet_direction)  # 添加数据包方向到特征数据
    feature_data.append(packet_message_type[0])  # 添加数据包消息类型到特征数据

    return feature_data  # 返回特征数据

'''
# 这段代码的主要任务是处理每个标签的网络流量数据，从中提取特征数据，并把它们保存为文件，以便后续进行模型训练。
def process_one_label(session_pcap_path, key, payload_length, payload_packet, samples, label_id, start_index=76):
    # 处理单个标签数据的函数
    result = {  # 初始化结果字典
        "samples": 0,  # 样本数量
        "datagram": {},  # 数据报文
        "length": {},  # 长度信息
        "time": {},  # 时间信息
        "direction": {},  # 方向信息
        "message_type": {}  # 消息类型
    }

    target_all_files = [x[0] + "/" + y for x in [(p, f) for p, d, f in os.walk(session_pcap_path[key])] for y in
                        x[1]]  # 获取目标目录下所有文件路径
    # for f in target_all_files:  # 注释掉的代码：遍历所有文件
    #     file_size = float(size_format(os.path.getsize(pcap_split_path+"splitcap" + "/" + dir + "/" + f)))  # 注释掉的代码：获取文件大小
    #     if file_size>  # 注释掉的代码：条件判断
    label_count = label_id[key]  # 获取标签计数
    if len(target_all_files) > samples[label_count]:  # 如果文件数量大于样本数量
        random.seed(10)  # 设置随机种子
        r_files = random.sample(target_all_files, samples[label_count])  # 随机抽样文件
    else:  # 如果文件数量小于等于样本数量
        r_files = target_all_files  # 使用所有文件
    for r_f in r_files:  # 遍历选中的文件
        try:  # 尝试执行
            feature_data = get_feature_flow(r_f, select_packet_len=payload_length, packets_num=payload_packet,
                                            start_index=start_index)  # 提取特征数据
        except:  # 捕获异常
            feature_data = -1  # 设置特征数据为错误代码

        if feature_data == -1:  # 如果特征数据提取失败
            continue  # 继续下一个文件

        result["samples"] += 1  # 样本数量加1
        if len(result["datagram"].keys()) > 0:  # 如果结果字典中已有数据
            result["datagram"][str(result["samples"])] = feature_data[0]  # 添加数据报文
            result["length"][str(result["samples"])] = \
                feature_data[1]  # 添加长度信息
            result["time"][str(result["samples"])] = \
                feature_data[2]  # 添加时间信息
            result["direction"][str(result["samples"])] = \
                feature_data[3]  # 添加方向信息
            result["message_type"][str(result["samples"])] = \
                feature_data[4]  # 添加消息类型
        else:  # 如果结果字典中没有数据
            result["datagram"]["1"] = feature_data[0]  # 添加第一个数据报文
            result["length"]["1"] = feature_data[1]  # 添加第一个长度信息
            result["time"]["1"] = feature_data[2]  # 添加第一个时间信息
            result["direction"]["1"] = feature_data[3]  # 添加第一个方向信息
            result["message_type"]["1"] = feature_data[4]  # 添加第一个消息类型

        # ✅ 修复：使用当前目录下的 temp 文件夹
        temp_dir = "./temp/"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

        with open(temp_dir + key, 'wb') as f:
            pickle.dump(result, f)
'''


def process_one_label(session_pcap_path, key, payload_length, payload_packet, samples, label_id, start_index=76):
    # 处理单个标签数据的函数
    result = {
        "samples": 0,
        "datagram": {},
        "length": {},
        "time": {},
        "direction": {},
        "message_type": {}
    }

    # 获取目标目录下所有文件路径
    # 增加安全性检查，确保目录存在
    current_path = session_pcap_path[key]
    if not os.path.exists(current_path):
        print(f"[Warning] Path not found for key {key}: {current_path}")
        target_all_files = []
    else:
        target_all_files = [x[0] + "/" + y for x in [(p, f) for p, d, f in os.walk(current_path)] for y in x[1]]

    label_count = label_id[key]

    # 随机采样逻辑
    if len(target_all_files) > samples[label_count]:
        random.seed(10)
        r_files = random.sample(target_all_files, samples[label_count])
    else:
        r_files = target_all_files

    # 遍历处理文件
    for r_f in r_files:
        try:
            # 提取特征
            feature_data = get_feature_flow(r_f, select_packet_len=payload_length, packets_num=payload_packet,
                                            start_index=start_index)
        except Exception as e:
            # print(f"Error processing {r_f}: {e}")
            feature_data = -1

        if feature_data == -1:
            continue

        # 如果成功提取，写入 result
        result["samples"] += 1
        idx_str = str(result["samples"])

        # 存入数据
        result["datagram"][idx_str] = feature_data[0]
        result["length"][idx_str] = feature_data[1]
        result["time"][idx_str] = feature_data[2]
        result["direction"][idx_str] = feature_data[3]
        result["message_type"][idx_str] = feature_data[4]

    # [关键修复]：文件保存操作必须在 for 循环之外！
    # 无论是否提取到样本，都要生成这个文件，防止 KeyError
    temp_dir = "./temp/"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)

    try:
        with open(os.path.join(temp_dir, key), 'wb') as f:
            pickle.dump(result, f)
    except Exception as e:
        print(f"[Error] Failed to save temp file for {key}: {e}")


# 使用多进程处理每个标签的网络流量数据，并通过process_one_label提取其中的特征。
# 最后将所有标签的特征数据存储到 JSON 文件 中，作为模型训练的数据集。
def generation_multiP(pcap_path, samples, dataset_save_path, payload_length=64, payload_packet=5, start_index=76):
    # 多进程生成数据的函数
    dataset = {}
    label_name_list = []
    session_pcap_path = {}

    if os.path.exists(pcap_path):
        # 获取所有非隐藏文件夹
        dirs = [d for d in os.listdir(pcap_path) if os.path.isdir(os.path.join(pcap_path, d)) and not d.startswith('.')]

        for dir_name in dirs:
            if dir_name.strip() != dir_name:
                print(f"[Warning] Found dirty folder in Step 2: '{dir_name}'. Skipping.")
                continue

            label_name_list.append(dir_name)
            session_pcap_path[dir_name] = os.path.join(pcap_path, dir_name)
    else:
        print(f"[Error] Pcap path not found: {pcap_path}")
        return

    print("label number: ", len(label_name_list))

    label_id = {}
    for index in range(len(label_name_list)):
        label_id[label_name_list[index]] = index
    for key in label_id.keys():
        print(key, label_id[key])

    print("\nBegin to generate features.")
    pbar = tqdm(total=len(session_pcap_path.keys()))
    pbar.set_description('generate features')
    update = lambda *args: pbar.update()

    pool = mp.Pool(min(120, len(label_name_list)))
    for key in session_pcap_path.keys():
        pool.apply_async(process_one_label,
                         (session_pcap_path, key, payload_length, payload_packet, samples, label_id, start_index),
                         callback=update)
    pool.close()
    pool.join()

    # --- [关键修复] 合并逻辑开始 (去掉了注释) ---
    print("\n[Step 2.5] Merging temp files into dataset.json...")

    if not os.path.exists("./temp/"):
        print("[Error] Temp directory not found! No features were generated.")
        return

    # 1. 读取临时文件
    for key in os.listdir("./temp/"):
        try:
            # 过滤掉非预期文件
            if key not in label_id:
                continue

            temp_file_path = os.path.join("./temp/", key)
            with open(temp_file_path, 'rb') as f:
                result = pickle.load(f)

            # 将结果存入 dataset 字典
            dataset[label_id[key]] = result

            # 删除临时文件
            os.remove(temp_file_path)
        except Exception as e:
            print(f"[Warning] Failed to merge temp file {key}: {e}")

    # 2. 统计数据并填补空缺 (防止 KeyError)
    all_data_number = 0
    for index in range(len(label_name_list)):
        current_name = label_name_list[index]
        current_id = label_id[current_name]

        if current_id in dataset:
            count = dataset[current_id]["samples"]
            all_data_number += count
            print(f"Label {current_id} ({current_name}): {count} samples")
        else:
            # 如果某个类别没有生成任何样本，必须初始化一个空结构，否则 Step 3 会报错
            print(f"[Warning] Label {current_id} ({current_name}): 0 samples (Empty)")
            dataset[current_id] = {
                "samples": 0,
                "datagram": {}, "length": {}, "time": {}, "direction": {}, "message_type": {}
            }

    print(f"Total samples collected: {all_data_number}")

    # 3. 保存 dataset.json
    if not os.path.exists(dataset_save_path):
        os.makedirs(dataset_save_path)

    json_path = os.path.join(dataset_save_path, "dataset.json")
    try:
        with open(json_path, "w") as f:
            json.dump(dataset, fp=f, ensure_ascii=False, indent=4)
        print(f"Successfully saved: {json_path}")
    except Exception as e:
        print(f"[Error] Failed to write dataset.json: {e}")

# 将 .pcapng 文件转换为 .pcap 格式（如果需要），并将其按类别分割。
# 对每个分割后的 .pcap 文件进行清理：删除小于 2KB 的文件。对大于 10MB 的文件重新分割。删除流量文件数少于 10 的类别。
# 生成清理过的数据集，其中每个类别至少有 10 个流量文件
'''def convert_splitcap(pcapng_path, pcap_path, pcap_split_path, is_pcap_label=False):
    # 转换和分割pcap文件的函数
    # pcapng_path: the path of pcapng files (if the traffic is the pacp type, pcapng_path = pcap_path)
    # pcap_path: the path of pcap files
    # pcap_split_path: the path of splited pcap files
    # pcapng to pcap
    if not os.listdir(pcap_path):  # 如果pcap路径为空
        for parent, dirs, files in os.walk(pcapng_path):  # 遍历pcapng路径
            for file in files:  # 遍历文件
                cmd = "editcap -F pcap %s %s"  # 定义转换命令
                command = cmd % (parent + "/" + file, pcap_path + "/" + file)  # 构建完整命令
                os.system(command)  # 执行命令
    # split pcap
    label_name_list = []  # 初始化标签名称列表
    for parent, dirs, files in os.walk(pcap_path):  # 遍历pcap路径
        if len(dirs) == 0:  # 如果没有子目录
            for file in files:  # 遍历文件
                os.system(f"mkdir {pcap_path + file[:-5]}")  # 创建目录
                os.system(f"mv {pcap_path + file} {pcap_path + file[:-5]}")  # 移动文件
                label_name_list.append(file.split(".")[-2])  # 添加标签名称
        else:  # 如果有子目录
            label_name_list.extend(dirs)  # 添加目录名到标签名称列表
        break  # 跳出循环
    print(len(label_name_list))  # 打印标签数量
    for dir in label_name_list:  # 遍历标签名称列表
        for p, dd, ff in os.walk(parent + "/" + dir):  # 遍历标签目录
            for file in ff:  # 遍历文件
                if is_pcap_label:  # 如果使用pcap标签
                    output_path = split_cap(pcap_split_path, p + "/", file, pcap_label=dir)  # 分割pcap文件并使用标签
                else:  # 如果不使用pcap标签
                    output_path = split_cap(pcap_split_path, p + "/", file)  # 分割pcap文件
    # remove small pcap and split again big pcap
    for p, dd, ff in os.walk(pcap_split_path + "splitcap"):  # 遍历分割后的pcap路径
        for dir in dd:  # 遍历目录
            for _, _, ff in os.walk(pcap_split_path + "splitcap" + "/" + dir):  # 遍历文件
                for f in ff:  # 遍历文件
                    file_size = float(
                        size_format(os.path.getsize(pcap_split_path + "splitcap" + "/" + dir + "/" + f)))  # 获取文件大小
                    # 2KB
                    if file_size < 2:  # remove small pcap  # 如果文件大小小于2KB
                        os.remove(pcap_split_path + "splitcap" + "/" + dir + "/" + f)  # 删除小文件
                        # print("remove sample: %s for its size is less than 2 KB." % (pcap_split_path+"splitcap" + "/" + dir + "/" + f))  # 注释掉的调试信息
                    if file_size > 10240:  # 10MB  split again big pcap  # 如果文件大小大于10MB
                        print("bigger than 10MB")  # 打印信息
                        cmd = "editcap -i 300 {} {}".format(pcap_split_path + "splitcap" + "/" + dir + "/" + f,
                                                            pcap_split_path + "splitcap" + "/" + dir + "/" + f)  # 定义分割命令
                        os.system(cmd)  # 执行命令
                        os.system("rm {}".format(pcap_split_path + "splitcap" + "/" + dir + "/" + f))  # 删除原文件
                break  # 跳出循环
        break  # 跳出循环
    # remove class that has less flow
    all_flows = []  # 初始化所有流列表
    for p, dd, ff in os.walk(pcap_split_path + "splitcap"):  # 遍历分割后的pcap路径
        for dir in dd:  # 遍历目录
            for _, _, ff in os.walk(pcap_split_path + "splitcap" + "/" + dir):  # 遍历文件
                print(dir, len(ff))  # 打印目录名和文件数量
                if len(ff) < 10:  # 如果文件数量少于10
                    shutil.rmtree(pcap_split_path + "splitcap" + "/" + dir)  # 删除目录
                    print("remove class: %s for its flow size is less than 10." % (
                            pcap_split_path + "splitcap" + "/" + dir))  # 打印删除信息
                else:  # 如果文件数量大于等于10
                    all_flows.append(len(ff))  # 添加文件数量到所有流列表
        break  # 跳出循环
    print("all flows: ", sum(all_flows), len(all_flows))  # 打印总流数量和类别数量
'''


def convert_splitcap(pcapng_path, pcap_path, pcap_split_path, is_pcap_label=False):
    print("--- Starting Convert & Split ---")

    # 0. 预处理：强制修复 pcap_path 下已存在的“脏”文件夹名
    # 如果磁盘上已经有 "vpn-voip "，这步会强制把它改成 "vpn-voip"
    if os.path.exists(pcap_path):
        for d in os.listdir(pcap_path):
            src_path = os.path.join(pcap_path, d)
            if os.path.isdir(src_path):
                # 去除首尾空格，并去除中间可能存在的空格(可选，视情况而定)
                clean_name = d.strip()
                if clean_name != d:
                    dst_path = os.path.join(pcap_path, clean_name)
                    print(f"[Auto-Fix] Renaming dirty folder: '{d}' -> '{clean_name}'")
                    try:
                        # 如果目标文件夹已存在（比如同时有 vpn 和 'vpn '），则合并内容
                        if os.path.exists(dst_path):
                            for f in os.listdir(src_path):
                                shutil.move(os.path.join(src_path, f), os.path.join(dst_path, f))
                            os.rmdir(src_path)
                        else:
                            os.rename(src_path, dst_path)
                    except Exception as e:
                        print(f"Warning: Failed to rename {d}: {e}")

    # 1. pcapng to pcap
    if not os.path.exists(pcap_path) or not os.listdir(pcap_path):
        for parent, dirs, files in os.walk(pcapng_path):
            for file in files:
                if file.endswith('.pcapng'):
                    cmd = "editcap -F pcap \"%s\" \"%s\"" % (
                        os.path.join(parent, file),
                        os.path.join(pcap_path, file)
                    )
                    os.system(cmd)

    # 2. split pcap (整理文件夹)
    # 此时磁盘上的文件夹应该都是干净的了
    label_name_list = []
    if not os.path.exists(pcap_path):
        os.makedirs(pcap_path)

    for parent, dirs, files in os.walk(pcap_path):
        # 即使有子文件夹，我们也再次检查是否有散落的文件需要归类
        for file in files:
            if not file.endswith('.pcap'): continue

            # [核心逻辑] 获取干净的标签名
            # 假设文件名 "vpn-voip .pcap" -> split后 "vpn-voip " -> strip后 "vpn-voip"
            clean_label = file.split(".")[-2].strip()

            # 安全检查：如果标签为空，跳过
            if not clean_label: continue

            target_dir = os.path.join(pcap_path, clean_label)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # 移动文件
            source_file = os.path.join(parent, file)
            dest_file = os.path.join(target_dir, file)

            # 防止自己移动到自己
            if os.path.dirname(source_file) != target_dir:
                try:
                    shutil.move(source_file, dest_file)
                except Exception as e:
                    pass

            if clean_label not in label_name_list:
                label_name_list.append(clean_label)

        # 收集已经存在的（无论是刚创建的还是原本有的）干净文件夹名
        for d in dirs:
            clean_d = d.strip()
            if clean_d not in label_name_list:
                label_name_list.append(clean_d)
        break

    print(f"Verified Categories: {len(label_name_list)}")

    # 3. 执行 SplitCap
    # 重新读取目录，确保准确
    current_dirs = [d for d in os.listdir(pcap_path) if os.path.isdir(os.path.join(pcap_path, d))]

    for dir_name in current_dirs:
        # dir_name 必然是无空格的
        full_dir_path = os.path.join(pcap_path, dir_name)

        for p, dd, ff in os.walk(full_dir_path):
            for file in ff:
                if not file.endswith('.pcap'): continue

                # SplitCap 输出路径，确保 pcap_label 是干净的
                if is_pcap_label:
                    split_cap(pcap_split_path, p + "/", file, pcap_label=dir_name)
                else:
                    split_cap(pcap_split_path, p + "/", file)

    # 4. 清理小文件和二次切割
    splitcap_root = os.path.join(pcap_split_path, "splitcap")
    if not os.path.exists(splitcap_root):
        print("Warning: splitcap dir not created.")
        return

    # 遍历 splitcap 输出目录，这里也可能产生带空格的文件夹（如果 SplitCap 行为怪异）
    # 我们再次进行一次清理
    for d in os.listdir(splitcap_root):
        d_path = os.path.join(splitcap_root, d)
        if os.path.isdir(d_path):
            clean_d = d.strip()
            if clean_d != d:
                print(f"[Auto-Fix SplitCap Output] Renaming {d} -> {clean_d}")
                new_d_path = os.path.join(splitcap_root, clean_d)
                # 如果目标存在，先移除目标再重命名(暴力覆盖)或者合并
                if os.path.exists(new_d_path):
                    shutil.rmtree(new_d_path)
                os.rename(d_path, new_d_path)

    # 处理文件大小
    for p, dd, ff in os.walk(splitcap_root):
        for dir in dd:
            dir_path = os.path.join(splitcap_root, dir)
            for f in os.listdir(dir_path):
                file_full_path = os.path.join(dir_path, f)
                try:
                    file_size = float(size_format(os.path.getsize(file_full_path)))
                    if file_size < 2:
                        os.remove(file_full_path)
                    elif file_size > 10240:
                        cmd = f"editcap -i 300 \"{file_full_path}\" \"{file_full_path}\""
                        os.system(cmd)
                        if os.path.exists(file_full_path):
                            os.remove(file_full_path)
                except:
                    continue
        break

        # 5. 删除样本过少的类别
    all_flows = []
    dirs_to_remove = []
    if os.path.exists(splitcap_root):
        categories = os.listdir(splitcap_root)
        for dir in categories:
            dir_path = os.path.join(splitcap_root, dir)
            if not os.path.isdir(dir_path): continue

            files = os.listdir(dir_path)
            if len(files) < 10:
                dirs_to_remove.append(dir_path)
                print(f"Removing class {dir} (flows < 10)")
            else:
                all_flows.append(len(files))

        for d in dirs_to_remove:
            shutil.rmtree(d)

    print(f"Total flows: {sum(all_flows)}, Categories: {len(all_flows)}")
def dataset_extract(dataset_save_path, features, dataset_level="flow"):
    print("read dataset from json file.")
    with open(dataset_save_path + "/dataset.json", "r") as f:
        dataset = json.load(f)

    # [修复2] 动态获取类别数量，替代 _category
    category_count = len(dataset.keys())
    dataset_statistic = [0] * category_count

    data_all = []
    for app_label in dataset.keys():
        if not app_label.isdigit(): continue
        label_idx = int(app_label)

        # 安全检查
        if label_idx >= category_count: continue

        # 获取样本数
        sample_count = len(dataset[app_label]["length"])

        for index_sample in range(sample_count):
            x = []
            for feature in features:
                # 注意 json key 是字符串 "1", "2"...
                val = dataset[app_label][feature].get(str(index_sample + 1))
                x.append(val)
            x.append(label_idx)
            dataset_statistic[label_idx] += 1
            data_all.append(x)

    data = pd.DataFrame(data_all, columns=features + ['label'])

    print("category flow")
    for index in range(len(dataset_statistic)):
        print("%s\t%d" % (index, dataset_statistic[index]))
    print("all\t%d" % (sum(dataset_statistic)))

    if len(data) < 5:
        print("Data too small to split!")
        return

    # 分割数据集
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=41, stratify=data["label"])
    data_val, data_test = train_test_split(data_test, test_size=0.5, random_state=42, stratify=data_test["label"])

    print("label number of train: {}, val: {}, test: {}.".format(len(data_train['label'].value_counts()),
                                                                 len(data_val['label'].value_counts()),
                                                                 len(data_test['label'].value_counts())))
    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    # 路径修复
    output_dir = os.path.join(dataset_save_path, "dataset")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存 tsv
    write_dataset_tsv(data_train['datagram'], data_train['label'], output_dir + "/", "train")
    write_dataset_tsv(data_test['datagram'], data_test['label'], output_dir + "/", "test")
    write_dataset_tsv(data_val['datagram'], data_val['label'], output_dir + "/", "valid")

    print("finish generating pre-train's datagram dataset.\nPlease check in %s" % output_dir)

"""
def dataset_extract(dataset_save_path, features):
    # 从数据集中提取特征的函数

    print("read dataset from json file.")  # 打印读取数据集信息
    with open(dataset_save_path + "/dataset.json", "r") as f:  # 打开数据集JSON文件
        dataset = json.load(f)  # 加载JSON数据

    dataset_statistic = [0] * _category  # 初始化数据集统计列表

    data_all = []  # 初始化所有数据列表
    for app_label in dataset.keys():  # 遍历数据集键值
        for index_sample in range(len(dataset[app_label]["length"])):  # 遍历样本索引
            x = []  # 初始化特征列表
            for feature in features:  # 遍历特征列表
                x.append(dataset[app_label][feature][str(index_sample + 1)])  # 添加特征值
            x.append(int(app_label))  # 添加标签
            dataset_statistic[int(app_label)] += 1  # 更新统计计数
            data_all.append(x)  # 添加到所有数据列表
    data = pd.DataFrame(data_all, columns=features + ['label'])  # 创建DataFrame

    print("category flow")  # 打印类别流量信息
    for index in range(len(dataset_statistic)):  # 遍历统计列表
        print("%s\t%d" % (index, dataset_statistic[index]))  # 打印类别和数量
    print("all\t%d" % (sum(dataset_statistic)))  # 打印总数量

    # split train set and test set
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=41, stratify=data["label"])  # 分割训练集和测试集
    # split validate set and test set
    data_val, data_test = train_test_split(data_test, test_size=0.5, random_state=42,
                                           stratify=data_test["label"])  # 分割验证集和测试集

    print("label number of train: {}, val: {}, test: {}.".format(len(data_train['label'].value_counts()),
                                                                 len(data_val['label'].value_counts()),
                                                                 len(data_test['label'].value_counts())))  # 打印各集合标签数量
    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    # === 修复开始：添加 "/" 分隔符 ===
    # 原始代码: dataset_save_path + "dataset/"
    # 修复代码: dataset_save_path + "/dataset/"

    output_dir = dataset_save_path + "/dataset/"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # save bytes to tsv
    write_dataset_tsv(data_train['datagram'], data_train['label'], output_dir, "train")
    write_dataset_tsv(data_test['datagram'], data_test['label'], output_dir, "test")
    write_dataset_tsv(data_val['datagram'], data_val['label'], output_dir, "valid")

    print("finish generating pre-train's datagram dataset.\nPlease check in %s" % output_dir)
    '''data_train = data_train.reset_index(drop=True)  # 重置训练集索引
    data_val = data_val.reset_index(drop=True)  # 重置验证集索引
    data_test = data_test.reset_index(drop=True)  # 重置测试集索引

    if not os.path.exists(dataset_save_path + "dataset/"):  # 如果数据集目录不存在
        os.mkdir(dataset_save_path + "dataset/")  # 创建数据集目录

    # save features to .pkl
    # with open(os.path.join(dataset_save_path + "dataset/", 'train.pkl'),"wb") as f:  # 注释掉的代码：保存训练集到pkl文件
    #     pickle.dump(data_train,f)  # 注释掉的代码：序列化训练集
    # with open(os.path.join(dataset_save_path + "dataset/", 'test.pkl'),"wb") as f:  # 注释掉的代码：保存测试集到pkl文件
    #     pickle.dump(data_test,f)  # 注释掉的代码：序列化测试集
    # with open(os.path.join(dataset_save_path + "dataset/", 'valid.pkl'),"wb") as f:  # 注释掉的代码：保存验证集到pkl文件
    #     pickle.dump(data_val,f)  # 注释掉的代码：序列化验证集

    # save bytes to tsv
    write_dataset_tsv(data_train['datagram'], data_train['label'], dataset_save_path + "dataset/",
                      "train")  # 保存训练集到TSV文件
    write_dataset_tsv(data_test['datagram'], data_test['label'], dataset_save_path + "dataset/", "test")  # 保存测试集到TSV文件
    write_dataset_tsv(data_val['datagram'], data_val['label'], dataset_save_path + "dataset/", "valid")  # 保存验证集到TSV文件
    print(
        "finish generating pre-train's datagram dataset.\nPlease check in %s" % dataset_save_path + "dataset/")  # 打印完成信息
    '''
"""

# 基于文件内容的整体数据增强，主要处理表格数据或文本数据，修改字段后生成新的增强样本。
'''
def enhance_based_tsv(path, filename, new_file_prefix, enhance_factor=1):
    # 基于TSV文件进行数据增强的函数
    # path: the tsv path
    # filenmae: the name of tsv path
    # new_file_prefix: the prefix of enhanced tsv
    # enhance_factor: augmentation factor
    dataset = []  # 初始化数据集列表
    columns = {}  # 初始化列字典
    with open(path + filename, mode="r", encoding="utf-8") as f:  # 打开TSV文件
        for line_id, line in enumerate(f):  # 遍历文件行
            if line_id == 0:  # 如果是第一行（表头）
                for i, column_name in enumerate(line.strip().split("\t")):  # 遍历列名
                    columns[column_name] = i  # 记录列索引
                continue  # 继续下一行
            line = line[:-1].split("\t")  # 分割行数据
            tgt = int(line[columns["label"]])  # 获取标签
            text_a = line[columns["text_a"]]  # 获取文本数据
            text_list = text_a.split("[SEP]")[1:]  # 分割文本列表
            for _ in range(enhance_factor):  # 遍历增强因子
                # IPID:4, src:12, dst: 16, sport:20, dport:22, seq:24, ack:28
                IP, proto, first_forward_datagrams, first_backward_datagrams = None, None, None, None  # 初始化变量
                datagramss = []  # 初始化数据报文列表
                for i in range(len(text_list)):  # 遍历文本列表
                    pac = text_list[i]  # 获取数据包文本
                    datagrams = pac.split(" ")[1:-1]  # 分割数据报文
                    datagramss.append(datagrams)  # 添加到数据报文列表
                    if i == 0:  # 如果是第一个数据包
                        if datagrams[0][0] == "4":  # 如果是IPv4
                            IP = 4  # 设置IP版本
                            if datagrams[9][:2] == "06":
                                proto = 6  # 如果是TCP
                            elif datagrams[9][:2] == "11":
                                proto = 17  # 如果是UDP
                        elif datagrams[0][0] == "6":  # 如果是IPv6
                            IP = 6  # 设置IP版本
                            if datagrams[6][:2] == "06":
                                proto = 6  # 如果是TCP
                            elif datagrams[6][:2] == "11":
                                proto = 17  # 如果是UDP
                        src = datagrams[12:16]  # 获取源地址
                        first_forward_datagrams = datagrams  # 记录第一个正向数据报文
                    if datagrams[12:16] != src and first_backward_datagrams is None:  # 如果是第一个反向数据包
                        first_backward_datagrams = datagrams  # 记录第一个反向数据报文

                if IP == None or proto == None:  # 如果无法确定IP版本或协议
                    print(line)  # 打印行数据
                    # return  # 返回
                    continue

                if IP == 4:  # 如果是IPv4
                    rsrc = random_field(32)  # 生成随机源IP
                    rdst = random_field(32)  # 生成随机目标IP
                    rsrcid = random_field(16)  # 生成随机源ID
                    rdstid = random_field(16)  # 生成随机目标ID
                elif IP == 6:  # 如果是IPv6
                    print("IPV6 is waiting to process")  # 打印提示信息
                    continue  # 继续下一个循环
                if proto == 6:  # 如果是TCP
                    rsrcp = random_field(16)  # 生成随机源端口
                    rdstp = random_field(16)  # 生成随机目标端口
                    rsrcseq = random_field(32)  # 生成随机源序列号
                    rdstseq = random_field(32)  # 生成随机目标序列号
                elif proto == 17:  # 如果是UDP
                    rsrcp = random_field(16)  # 生成随机源端口
                    rdstp = random_field(16)  # 生成随机目标端口

                forward_4tstr = hex(rsrc)[2:].zfill(8) + hex(rdst)[2:].zfill(8) + hex(rsrcp)[2:].zfill(4) + hex(rdstp)[
                                                                                                            2:].zfill(
                    4)  # 构建正向四元组字符串
                backward_4tstr = hex(rdst)[2:].zfill(8) + hex(rsrc)[2:].zfill(8) + hex(rdstp)[2:].zfill(4) + hex(rsrcp)[
                                                                                                             2:].zfill(
                    4)  # 构建反向四元组字符串
                
                srcipid = int(first_forward_datagrams[4], 16)  # 获取源IPID
                if first_backward_datagrams is not None:  # 如果有反向数据报文
                    dstipid = int(first_backward_datagrams[4], 16)  # 获取目标IPID
                srcseq = int(first_forward_datagrams[24] + first_forward_datagrams[26], 16)  # 获取源序列号
                srcack = int(first_forward_datagrams[28] + first_forward_datagrams[30], 16)  # 获取源确认号
                if first_backward_datagrams is not None:  # 如果有反向数据报文
                    dstseq = int(first_backward_datagrams[24] + first_backward_datagrams[26], 16)  # 获取目标序列号
                elif srcack != 0:  # 如果源确认号不为0
                    dstseq = srcack  # 使用源确认号作为目标序列号
                else:  # 如果无法确定目标序列号
                    print("cant process dstseq...")  # 打印错误信息
                # print(hex(srcseq),hex(dstseq),hex(rsrcseq),hex(rdstseq))  # 注释掉的调试信息

                # print(forward_4tstr, backward_4tstr)    # 注释掉的调试信息
                for i in range(len(datagramss)):  # 遍历数据报文列表
                    # print("------")  # 注释掉的调试信息
                    # print(datagramss[i])  # 注释掉的调试信息
                    if datagramss[i][12:16] == src:  # forward  # 如果是正向数据包
                        datagramss[i][11] = datagramss[i][11][:2] + forward_4tstr[:2]  # 更新IP头校验和
                        cc = 12  # 初始化位置计数器
                        for elm in bigram_generation(forward_4tstr, token_len=len(forward_4tstr) / 2).split(" ")[
                                   :-1]:  # 遍历四元组bigram
                            datagramss[i][cc] = elm  # 更新数据报文
                            cc += 1  # 位置计数器加1
                        datagramss[i][cc] = forward_4tstr[-2:] + datagramss[i][cc][2:4]  # 更新最后一个元素
                        # handle IPID
                        if IP == 4:  # 如果是IPv4
                            if srcipid != 0:  # 如果源IPID不为0
                                temp = hex((int(datagramss[i][4], 16) - srcipid + rsrcid) % (2 ** 16))[2:].zfill(
                                    4)  # 计算新的IPID
                                datagramss[i][4] = temp  # 更新IPID
                                datagramss[i][3] = datagramss[i][3][:2] + temp[:2]  # 更新IP头相关字段
                                datagramss[i][5] = temp[2:] + datagramss[i][5][2:]  # 更新IP头相关字段
                        # handle seq
                        if proto == 6:  # 如果是TCP
                            tempsrcseq = hex(
                                (rsrcseq + int(datagramss[i][24] + datagramss[i][26], 16) - srcseq) % (2 ** 32))[
                                         2:].zfill(8)  # 计算新的序列号
                            datagramss[i][23] = datagramss[i][23][:2] + tempsrcseq[:2]  # 更新TCP头相关字段
                            cc = 24  # 设置位置计数器
                            for elm in bigram_generation(tempsrcseq, len(tempsrcseq) / 2).split(" ")[
                                       :-1]:  # 遍历序列号bigram
                                datagramss[i][cc] = elm  # 更新数据报文
                                cc += 1  # 位置计数器加1
                            datagramss[i][cc] = tempsrcseq[-2:] + datagramss[i][cc][2:4]  # 更新最后一个元素
                            # handle ack
                            if int(datagramss[i][28] + datagramss[i][30], 16) != 0 and dstseq:  # 如果确认号不为0且有目标序列号
                                tempsrcack = hex(
                                    (rdstseq + int(datagramss[i][28] + datagramss[i][30], 16) - dstseq) % (2 ** 32))[
                                             2:].zfill(8)  # 计算新的确认号
                                datagramss[i][27] = datagramss[i][27][:2] + tempsrcack[:2]  # 更新TCP头相关字段
                                cc = 28  # 设置位置计数器
                                for elm in bigram_generation(tempsrcack, len(tempsrcack) / 2).split(" ")[
                                           :-1]:  # 遍历确认号bigram
                                    datagramss[i][cc] = elm  # 更新数据报文
                                    cc += 1  # 位置计数器加1
                                datagramss[i][cc] = tempsrcack[-2:] + datagramss[i][cc][2:4]  # 更新最后一个元素

                    else:  # 如果是反向数据包
                        datagramss[i][11] = datagramss[i][11][:2] + backward_4tstr[:2]  # 更新IP头校验和
                        cc = 12  # 初始化位置计数器
                        for elm in bigram_generation(backward_4tstr, token_len=len(backward_4tstr) / 2).split(" ")[
                                   :-1]:  # 遍历四元组bigram
                            datagramss[i][cc] = elm  # 更新数据报文
                            cc += 1  # 位置计数器加1
                        datagramss[i][cc] = backward_4tstr[-2:] + datagramss[i][cc][2:4]  # 更新最后一个元素
                        # handle IPID
                        if IP == 4:  # 如果是IPv4
                            if dstipid != 0:  # 如果目标IPID不为0
                                temp = hex((int(datagramss[i][4], 16) - dstipid + rdstid) % (2 ** 16))[2:].zfill(
                                    4)  # 计算新的IPID
                                datagramss[i][4] = temp  # 更新IPID
                                datagramss[i][3] = datagramss[i][3][:2] + temp[:2]  # 更新IP头相关字段
                                datagramss[i][5] = temp[2:] + datagramss[i][5][2:]  # 更新IP头相关字段
                        # handle seq
                        if proto == 6:  # 如果是TCP
                            if dstseq:  # 如果有目标序列号
                                tempdstseq = hex(
                                    (rdstseq + int(datagramss[i][24] + datagramss[i][26], 16) - dstseq) % (2 ** 32))[
                                             2:].zfill(8)  # 计算新的序列号
                                datagramss[i][23] = datagramss[i][23][:2] + tempdstseq[:2]  # 更新TCP头相关字段
                                cc = 24  # 设置位置计数器
                                for elm in bigram_generation(tempdstseq, len(tempdstseq) / 2).split(" ")[
                                           :-1]:  # 遍历序列号bigram
                                    datagramss[i][cc] = elm  # 更新数据报文
                                    cc += 1  # 位置计数器加1
                                datagramss[i][cc] = tempdstseq[-2:] + datagramss[i][cc][2:4]  # 更新最后一个元素
                            # handle ack
                            if int(datagramss[i][28] + datagramss[i][30], 16) != 0:  # 如果确认号不为0
                                tempdstack = hex(
                                    (rsrcseq + int(datagramss[i][28] + datagramss[i][30], 16) - srcseq) % (2 ** 32))[
                                             2:].zfill(8)  # 计算新的确认号
                                datagramss[i][27] = datagramss[i][27][:2] + tempdstack[:2]  # 更新TCP头相关字段
                                cc = 28  # 设置位置计数器
                                for elm in bigram_generation(tempdstack, len(tempdstack) / 2).split(" ")[
                                           :-1]:  # 遍历确认号bigram
                                    datagramss[i][cc] = elm  # 更新数据报文
                                    cc += 1  # 位置计数器加1
                                datagramss[i][cc] = tempdstack[-2:] + datagramss[i][cc][2:4]  # 更新最后一个元素
                    # print(datagramss[i])  # 注释掉的调试信息

                newtext_a = ''  # 初始化新文本
                for i in range(len(datagramss)):  # 遍历数据报文列表
                    if newtext_a != '':  # 2024.4.23 add  # 如果不是第一个数据包
                        newtext_a += ' '  # 添加空格
                    newtext_a += '[SEP]'  # 添加分隔符
                    for j in range(len(datagramss[i])):  # 遍历数据报文元素
                        if newtext_a != '':  # 如果不是第一个元素
                            newtext_a += ' '  # 添加空格
                        newtext_a += datagramss[i][j]  # 添加数据报文元素

                dataset.append([newtext_a, tgt])  # 添加到数据集

    dataset = pd.DataFrame(dataset, columns=['datagram', 'label'])  # 创建DataFrame
    dataset = dataset.sample(frac=1)  # 随机打乱数据
    # print(dataset.head())  # 注释掉的调试信息
    write_dataset_tsv(dataset['datagram'], dataset['label'], path, new_file_prefix)  # 保存增强后的数据集到TSV文件
'''


# ... (文件的其他部分保持不变) ...

# ==========================================
# 修复协议判断逻辑 (修复 Unknown Protocol 问题)
# ==========================================
def enhance_based_tsv(path, filename, new_file_prefix, enhance_factor=1):
    dataset = []
    columns = {}

    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        print(f"Error: File {full_path} not found.")
        return

    print(f"Enhancing data from {filename} with factor {enhance_factor}...")

    # 调试计数器
    skip_count = 0
    success_count = 0
    error_reasons = {}

    with open(full_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue

            line_parts = line[:-1].split("\t")
            if len(line_parts) <= columns.get("label", 0) or len(line_parts) <= columns.get("text_a", 1):
                continue

            tgt = int(line_parts[columns["label"]])
            text_a = line_parts[columns["text_a"]]
            text_list = text_a.split("[SEP]")[1:]

            for _ in range(enhance_factor):
                IP, proto = None, None
                first_forward_datagrams = None
                first_backward_datagrams = None
                datagramss = []
                src = None

                try:
                    for i in range(len(text_list)):
                        pac = text_list[i]
                        datagrams = pac.split(" ")[1:-1]
                        datagramss.append(datagrams)

                        if i == 0:
                            if len(datagrams) < 20:
                                raise ValueError(f"Packet too short (len={len(datagrams)})")

                            # [关键修复] 使用切片 [:2] 来匹配 Bigram 的前两个字符
                            if datagrams[0][0] == "4":
                                IP = 4
                                if len(datagrams) > 9:
                                    # datagrams[9] 是第10个Bigram (Byte 9-10)
                                    # IPv4 Protocol 字段在 Byte 9
                                    if datagrams[9][:2] == "06":
                                        proto = 6
                                    elif datagrams[9][:2] == "11":
                                        proto = 17
                            elif datagrams[0][0] == "6":
                                IP = 6
                                if len(datagrams) > 6:
                                    # IPv6 Next Header 字段在 Byte 6
                                    if datagrams[6][:2] == "06":
                                        proto = 6
                                    elif datagrams[6][:2] == "11":
                                        proto = 17

                            # 获取源IP
                            if IP == 4 and len(datagrams) >= 16:
                                src = datagrams[12:16]
                                first_forward_datagrams = datagrams

                        if i > 0 and IP == 4 and len(datagrams) >= 16 and src is not None:
                            if datagrams[12:16] != src and first_backward_datagrams is None:
                                first_backward_datagrams = datagrams

                except Exception as e:
                    reason = str(e)
                    error_reasons[reason] = error_reasons.get(reason, 0) + 1
                    if skip_count < 5: print(f"[Debug] Skip line {line_id}: Parse error - {e}")
                    skip_count += 1
                    continue

                if IP is None:
                    reason = "Unknown IP version"
                    error_reasons[reason] = error_reasons.get(reason, 0) + 1
                    continue
                if proto is None:
                    reason = "Unknown Protocol"
                    error_reasons[reason] = error_reasons.get(reason, 0) + 1
                    continue
                if first_forward_datagrams is None:
                    reason = "No forward packet found"
                    error_reasons[reason] = error_reasons.get(reason, 0) + 1
                    continue

                required_len = 32
                if len(first_forward_datagrams) < required_len:
                    reason = f"First packet too short (<{required_len})"
                    error_reasons[reason] = error_reasons.get(reason, 0) + 1
                    continue

                # 数据增强逻辑
                try:
                    if IP == 4:
                        rsrc = random_field(32)
                        rdst = random_field(32)
                        rsrcid = random_field(16)
                        rdstid = random_field(16)
                    else:
                        continue

                    if proto == 6:
                        rsrcp = random_field(16)
                        rdstp = random_field(16)
                        rsrcseq = random_field(32)
                        rdstseq = random_field(32)
                    elif proto == 17:
                        rsrcp = random_field(16)
                        rdstp = random_field(16)
                    else:
                        continue

                    forward_4tstr = hex(rsrc)[2:].zfill(8) + hex(rdst)[2:].zfill(8) + hex(rsrcp)[2:].zfill(4) + hex(
                        rdstp)[2:].zfill(4)
                    backward_4tstr = hex(rdst)[2:].zfill(8) + hex(rsrc)[2:].zfill(8) + hex(rdstp)[2:].zfill(4) + hex(
                        rsrcp)[2:].zfill(4)

                    srcipid = int(first_forward_datagrams[4], 16)
                    dstipid = 0
                    if first_backward_datagrams:
                        dstipid = int(first_backward_datagrams[4], 16)

                    srcseq, srcack, dstseq = 0, 0, 0
                    if len(first_forward_datagrams) > 30:
                        srcseq = int(first_forward_datagrams[24] + first_forward_datagrams[26], 16)
                        srcack = int(first_forward_datagrams[28] + first_forward_datagrams[30], 16)

                    if first_backward_datagrams and len(first_backward_datagrams) > 26:
                        dstseq = int(first_backward_datagrams[24] + first_backward_datagrams[26], 16)
                    elif srcack != 0:
                        dstseq = srcack

                    for i in range(len(datagramss)):
                        current_pkt = datagramss[i]
                        if len(current_pkt) < 16: continue

                        is_forward = (current_pkt[12:16] == src)

                        if is_forward:
                            if len(current_pkt) > 12:
                                current_pkt[11] = current_pkt[11][:2] + forward_4tstr[:2]
                                cc = 12
                                parts = bigram_generation(forward_4tstr, token_len=len(forward_4tstr) / 2).split(" ")[
                                        :-1]
                                for elm in parts:
                                    if cc < len(current_pkt): current_pkt[cc] = elm
                                    cc += 1
                                if cc < len(current_pkt):
                                    current_pkt[cc] = forward_4tstr[-2:] + current_pkt[cc][2:4]

                            if IP == 4 and srcipid != 0 and len(current_pkt) > 5:
                                val = (int(current_pkt[4], 16) - srcipid + rsrcid) % (2 ** 16)
                                temp = hex(val)[2:].zfill(4)
                                current_pkt[4] = temp
                                current_pkt[3] = current_pkt[3][:2] + temp[:2]
                                current_pkt[5] = temp[2:] + current_pkt[5][2:]

                            if proto == 6 and len(current_pkt) > 24:
                                seq_val = int(current_pkt[24] + current_pkt[26], 16)
                                new_seq = (rsrcseq + seq_val - srcseq) % (2 ** 32)
                                tempsrcseq = hex(new_seq)[2:].zfill(8)
                                current_pkt[23] = current_pkt[23][:2] + tempsrcseq[:2]
                                cc = 24
                                seq_parts = bigram_generation(tempsrcseq, len(tempsrcseq) / 2).split(" ")[:-1]
                                for elm in seq_parts:
                                    if cc < len(current_pkt): current_pkt[cc] = elm
                                    cc += 1
                                if cc < len(current_pkt):
                                    current_pkt[cc] = tempsrcseq[-2:] + current_pkt[cc][2:4]

                                if len(current_pkt) > 30 and dstseq:
                                    ack_val = int(current_pkt[28] + current_pkt[30], 16)
                                    if ack_val != 0:
                                        new_ack = (rdstseq + ack_val - dstseq) % (2 ** 32)
                                        tempsrcack = hex(new_ack)[2:].zfill(8)
                                        current_pkt[27] = current_pkt[27][:2] + tempsrcack[:2]
                                        cc = 28
                                        ack_parts = bigram_generation(tempsrcack, len(tempsrcack) / 2).split(" ")[:-1]
                                        for elm in ack_parts:
                                            if cc < len(current_pkt): current_pkt[cc] = elm
                                            cc += 1
                                        if cc < len(current_pkt):
                                            current_pkt[cc] = tempsrcack[-2:] + current_pkt[cc][2:4]

                        else:
                            if len(current_pkt) > 12:
                                current_pkt[11] = current_pkt[11][:2] + backward_4tstr[:2]
                                cc = 12
                                parts = bigram_generation(backward_4tstr, token_len=len(backward_4tstr) / 2).split(" ")[
                                        :-1]
                                for elm in parts:
                                    if cc < len(current_pkt): current_pkt[cc] = elm
                                    cc += 1
                                if cc < len(current_pkt):
                                    current_pkt[cc] = backward_4tstr[-2:] + current_pkt[cc][2:4]

                            if IP == 4 and dstipid != 0 and len(current_pkt) > 5:
                                val = (int(current_pkt[4], 16) - dstipid + rdstid) % (2 ** 16)
                                temp = hex(val)[2:].zfill(4)
                                current_pkt[4] = temp
                                current_pkt[3] = current_pkt[3][:2] + temp[:2]
                                current_pkt[5] = temp[2:] + current_pkt[5][2:]

                            if proto == 6 and len(current_pkt) > 24 and dstseq:
                                seq_val = int(current_pkt[24] + current_pkt[26], 16)
                                new_seq = (rdstseq + seq_val - dstseq) % (2 ** 32)
                                tempdstseq = hex(new_seq)[2:].zfill(8)
                                current_pkt[23] = current_pkt[23][:2] + tempdstseq[:2]
                                cc = 24
                                seq_parts = bigram_generation(tempdstseq, len(tempdstseq) / 2).split(" ")[:-1]
                                for elm in seq_parts:
                                    if cc < len(current_pkt): current_pkt[cc] = elm
                                    cc += 1
                                if cc < len(current_pkt):
                                    current_pkt[cc] = tempdstseq[-2:] + current_pkt[cc][2:4]

                                if len(current_pkt) > 30:
                                    ack_val = int(current_pkt[28] + current_pkt[30], 16)
                                    if ack_val != 0:
                                        new_ack = (rsrcseq + ack_val - srcseq) % (2 ** 32)
                                        tempdstack = hex(new_ack)[2:].zfill(8)
                                        current_pkt[27] = current_pkt[27][:2] + tempdstack[:2]
                                        cc = 28
                                        ack_parts = bigram_generation(tempdstack, len(tempdstack) / 2).split(" ")[:-1]
                                        for elm in ack_parts:
                                            if cc < len(current_pkt): current_pkt[cc] = elm
                                            cc += 1
                                        if cc < len(current_pkt):
                                            current_pkt[cc] = tempdstack[-2:] + current_pkt[cc][2:4]

                except Exception as e:
                    reason = f"Modification Error: {e}"
                    error_reasons[reason] = error_reasons.get(reason, 0) + 1
                    if skip_count < 5: print(f"[Debug] Skip line {line_id}: {reason}")
                    skip_count += 1
                    continue

                newtext_a = ''
                for i in range(len(datagramss)):
                    if newtext_a != '': newtext_a += ' '
                    newtext_a += '[SEP]'
                    for j in range(len(datagramss[i])):
                        if newtext_a != '': newtext_a += ' '
                        newtext_a += datagramss[i][j]

                dataset.append([newtext_a, tgt])
                success_count += 1

    print(f"Total processed: {line_id}, Success: {success_count}, Skipped: {skip_count}")
    if error_reasons:
        print("Skip reasons summary:", error_reasons)

    if dataset:
        dataset = pd.DataFrame(dataset, columns=['datagram', 'label'])
        dataset = dataset.sample(frac=1)
        if not os.path.exists(path):
            os.makedirs(path)
        write_dataset_tsv(dataset['datagram'], dataset['label'], path, new_file_prefix)
        print(f"Successfully generated enhanced file: {new_file_prefix}_dataset.tsv with {len(dataset)} samples.")
    else:
        print("Warning: No samples were augmented.")