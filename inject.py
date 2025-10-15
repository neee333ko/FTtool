import copy
import random
import struct
import torch
from anytree import Node
from . import make_tree as cn
import numpy as np
import torch



def get_weight(convs,fixed_state_dict):
    weights=[]
    
    for i in range(len(convs)):
        fixed_layer = convs[i]
        # fixed_layer = fixed_layer[:fixed_layer.index('_')] + fixed_layer[fixed_layer.index('_') + 1:].replace('_', '.')
        weight = copy.deepcopy(fixed_state_dict[fixed_layer + '.weight'])
        weights.append((fixed_layer,weight))
    
    return weights


# 权重错误注入
def inject_weight(fixed_model,rate):
    fixed_state_dict = copy.deepcopy(fixed_model.state_dict())

    root = Node('root')
    cn.makeTree(fixed_model,root,"")

    convs = cn.get_fc_conv_leaf_name(fixed_model,root)
    weights=get_weight(convs, fixed_state_dict)

    for fixed_layer,weight in weights:
        faulty_weight = inject_MBF(copy.deepcopy(weight), rate)
        # diff_count = torch.sum(faulty_weight != weight).item()
        # print(f"Number of different elements: {diff_count}")
        fixed_state_dict[fixed_layer + '.weight'] = faulty_weight
   
    return fixed_state_dict


# 脉冲神经元饱和错误注入，调用该方法前不要reset模型状态
def inject_saturate(fixed_model, names, rates):
    entity_list = []
    v_shape_list = []
    
    for name in names:
        entity = cn.get_entity(fixed_model, name)
        index = name.split('.')[-1]
        entity = getattr(entity, index)
        entity_list.append(entity)
        
        v_shape = entity.v.shape
        v_ones = torch.ones(v_shape[1:], dtype=entity.v.dtype, device=entity.v.device)
        v_shape_list.append(v_ones)
        

    for i in range(len(entity_list)):
        num = int(v_shape_list[i].numel() * rates[i])  
        
        print("The saturate neruon number in ",names[i],"is: ",num)
        
        v_flat = v_shape_list[i].view(-1)
        total_elements = v_flat.numel()
        
        indices = torch.randperm(total_elements)[:num]
        
        v_flat[indices] = torch.finfo(v_flat.dtype).min
        
        v_shape_list[i] = v_flat.view(v_shape_list[i].shape)
        
        entity_list[i].v_threshold = v_shape_list[i]



# 脉冲神经元死亡错误注入，调用该方法前不要reset模型状态
def inject_dead(fixed_model, names, rates):
    entity_list = []
    v_shape_list = []
    

    for name in names:
        entity = cn.get_entity(fixed_model, name)
        index = name.split('.')[-1]
        entity = getattr(entity, index)
        entity_list.append(entity)
        
        
        v_shape = entity.v.shape
        v_ones = torch.ones(v_shape[1:], dtype=entity.v.dtype, device=entity.v.device)
        v_shape_list.append(v_ones)
        

    for i in range(len(entity_list)):
        num = int(v_shape_list[i].numel() * rates[i])  
        
        print("The dead neruon number in ",names[i],"is: ",num)
        
        v_flat = v_shape_list[i].view(-1)
        total_elements = v_flat.numel()
        
        indices = torch.randperm(total_elements)[:num]
        
        v_flat[indices] = torch.finfo(v_flat.dtype).max
        
        v_shape_list[i] = v_flat.view(v_shape_list[i].shape)
        
        entity_list[i].v_threshold = v_shape_list[i]
        



    
    

# Mutiple_bit_flip
def inject_MBF(weights,rate):
    count = 0
    size = 0
    weights_shape = weights.shape
    
    if len(weights_shape) == 2:
        print('是线形层')
        len1 = weights_shape[0]
        len2 = weights_shape[1]
        size = len2 * len1 * 32
        num = (int)(len1 * len2 * rate * 32)
        
        flip_bit = np.zeros((len1, len2, 32))
        
        while count < num:
            para1 = random.randint(0,len1 - 1)
            para2 = random.randint(0,len2 - 1)
            bit_num = random.randint(1,32)

            if flip_bit[para1][para2][bit_num - 1] == 0:
                weights[para1][para2] = inject_SBF(weights[para1][para2],bit_num)
                flip_bit[para1][para2][bit_num - 1] = 1
                count += 1

        
    if len(weights_shape) == 4:
        print('是卷积层')
        len1 = weights_shape[0]
        len2 = weights_shape[1]
        len3 = weights_shape[2]
        len4 = weights_shape[3]
        size = len2 * len1 * len3 * len4 * 32
        num = (int)(len1 * len2 * len3 * len4 * rate * 32)
        
        flip_bit = np.zeros((len1, len2, len3, len4, 32))
        
        while count < num:
            para1 = random.randint(0,len1 - 1)
            para2 = random.randint(0,len2 - 1)
            para3 = random.randint(0,len3 - 1)
            para4 = random.randint(0,len4 - 1)
            bit_num = random.randint(1,32)
            
            if flip_bit[para1][para2][para3][para4][bit_num - 1] == 0:
                flip_bit[para1][para2][para3][para4][bit_num - 1] = 1
                weights[para1][para2][para3][para4] = inject_SBF(weights[para1][para2][para3][para4],bit_num)
                count += 1
            
    print(f'注入错误个数 ：{count} 当前层总参数 {size}')
    
    return weights


# Single_bit_flip
def inject_SBF(weight,num):
    num -= 1
    floatweight = float_to_bin32(weight)
    num_str = list(floatweight)

    if num_str[num] == '0':
        num_str[num] = '1'
    else:
        num_str[num] = '0'

    return bin32_to_float(num_str)


def float_to_bin32(value):
    # 使用struct.pack将浮点数打包成字节串
    packed = struct.pack('!f', value)

    # 使用struct.unpack将字节串解包成整数
    # 然后使用bin转换为二进制字符串
    binary_representation = bin(struct.unpack('!I', packed)[0])[2:].zfill(32)

    # 在二进制字符串的每4位之间插入分隔符|
    formatted_binary = '|'.join(binary_representation[i:i + 4] for i in range(0, len(binary_representation), 4))

    # print(f"{formatted_binary}")
    
    return binary_representation

def bin32_to_float(binary_str):
    # 确保输入是32位的二进制字符串
    if len(binary_str) != 32 or not all(bit in '01' for bit in binary_str):
        raise ValueError("输入必须是32位的二进制字符串")

    # 将二进制字符串转换为整数
    num_str = ''.join(binary_str)
    int_value = int(num_str, 2)

    # 将整数打包成字节串，然后解包为浮点数
    float_result = struct.unpack('!f', struct.pack('!I', int_value))[0]

    return float_result


if __name__ == '__main__':
    pass
    
    