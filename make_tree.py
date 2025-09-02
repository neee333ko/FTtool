from anytree import Node
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.layer import Conv2d,Linear

# 判断是否有子模块
def is_not_basic_module(entity):
    class_name = entity.__class__.__name__
    
    if 'Linear' in class_name or 'IFNode' in class_name:
        return False
    count = 0
    for _, _ in enumerate(entity.named_children()):
        count = count + 1
        return True

    return False

# 造树时获取实体
def get_next_entity(model,name):    
    name = name.split('_')
    length = len(name)
    entity = model

    if hasattr(entity,name[length - 1]):
        entity = getattr(entity,name[length - 1])
    else:
        return None
    
    return entity

# 获取叶子节点的父节点实体
def get_entity(model,name):
    name = name.split('_')
    length = len(name)
    entity = model

    for i in range(length-1):
        entity = getattr(entity,name[i])
    
    return entity

# 递归造树，深度优先, 有个前提是model的属性名不能有下划线，因此spikingjelly的模型会报错, 
# 建议把spikingjelly模型的属性名改成不带下划线的
def makeTree(model,node,name): 
    name_list = []

    # 获取同一层兄弟节点
    for _, tmp in enumerate(model.named_children()):
        name_list.append(tmp[0])
        
    for i in range(len(name_list)):
        if name == "":
            node_name = name_list[i]
        else:
            node_name = name+'_'+name_list[i]
            
        cur_node = Node(node_name,parent=node)

        entity = get_next_entity(model,node_name)
        
        
        if entity == None:
            return

        if is_not_basic_module(entity):
            makeTree(entity,cur_node,node_name)
            


# 获取脉冲神经元列表
def get_neuron_leaf_name(model,root):
    acts = []
    leaves = root.leaves
    
    print(leaves)
    
    for node in leaves:
        entity = get_entity(model,node.name)
        index = node.name.split('_')
        index = index[-1]
        entity = getattr(entity, index)
        
        if type(entity)  in (neuron.IFNode ,neuron.LIFNode):
            acts.append(node.name)
            
    return acts

# 获取全连接层，卷积层列表
def get_fc_conv_leaf_name(model,root):
    acts = []
    leaves = root.leaves

    
    for node in leaves:
        entity = get_entity(model,node.name)
        index = node.name.split('_')
        index = index[-1]
        entity = getattr(entity, index)

        if type(entity)  in (Conv2d ,Linear):
            acts.append(node.name)
            
    return acts





# # 更换神经元
# def replace_neuron_in_pytorch(model, acts):  
#     length = len(acts)
#     new_model = model

#     for i in range(length):
#         index = acts[i].split('_')[-1]
#         entity = get_entity(new_model, acts[i])
#         setattr(entity, index, NewIFNode())
        
#     return new_model  



if __name__ == '__main__':
    pass
    


    




    
    





