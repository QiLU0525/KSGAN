import pandas as pd
import numpy as np
import json
from collections import namedtuple
import os
from scipy.sparse import lil_matrix
import random

Data = namedtuple('Data', ['size', 'user_entity', 'center_entities', 'history_mask','history_mask_3d', 'label'])

#以 xxx_entity 命名的都是embedding矩阵的下标

def padding(padded_index, max_length, array):
    # for cross attention
    mask = np.zeros((len(array),max_length))
    mask_3d = np.zeros((len(array), max_length, max_length))

    for i in range(len(array)):
        arr1 = array[i]
        none_zero = len(arr1)
        for j in range(none_zero):
            mask[i][j] = 1
            for k in range(none_zero):
                mask_3d[i][j][k] = 1

        for j in range(max_length-none_zero):
            array[i].append(padded_index)

    array = np.array(array)
    return array, mask.tolist(), mask_3d.tolist()

def read(args, file):
    #读训练文件和测试文件
    df = pd.read_csv(file, sep='\t', header=None, names=['user_entity', 'history_entities', 'candidate_entities', 'label'])
    print(file + ' shape: ',df.shape)

    unpadded_history = df['history_entities'].map(lambda x: json.loads(x))

    # history_mask 是 [-1,10] or [-1,10,10] 的 np array
    padded_history, hist_mask, hist_mask_3d = padding(padded_index='padding_position',
                            max_length=args.max_click_history, array=unpadded_history)

    df['history_entities'] = padded_history
    df['history_mask'] = hist_mask
    df['history_mask_3d'] = hist_mask_3d

    center_entities = []

    for i in range(df.shape[0]):
        li = df.iloc[i, 1]
        li.append(df.iloc[i, 2])
        center_entities.append(li)

    df['center_entities'] = center_entities
    return df

def read_entity2id(file):
    # Mendeley 读出来只有791991个key value键值对，因为 Uyun S 这个实体（author）出现了两次，不过不影响
    entity_dict = {}
    ent_id = pd.read_csv(file,header=None,sep = '\t')
    for i in range(ent_id.shape[0]):
        key = ent_id.iloc[i,0].strip()
        value = ent_id.iloc[i,1]
        entity_dict[key] = value
    return entity_dict

def load_sparse_test(args):
    print('loading sparse tests...')
    entity2index = read_entity2id('../BiQUE/data/' + args.data_source + '/ent_id')
    entity2index['padding_position'] = len(entity2index)
    test_groups = {}
    # 默认是 Mendeley
    # group_names = ['test_low2','test_low3','test_low5','test_low11']
    # group_names = ['test_1_inter', 'test_2_inter', 'test_34_inter', 'test_567_inter']
    group_names = ['test_1_inter', 'test_2_inter', 'test_3_inter', 'test_4_inter', 'test_5_inter']
    if args.data_source == 'Mendeley':
        test_1 = read(args, '../Dataset/{}/sparse/{}.txt'.format(args.data_source, group_names[0]))
        test_2 = read(args, '../Dataset/{}/sparse/{}.txt'.format(args.data_source, group_names[1]))
        test_3 = read(args, '../Dataset/{}/sparse/{}.txt'.format(args.data_source, group_names[2]))
        test_4 = read(args, '../Dataset/{}/sparse/{}.txt'.format(args.data_source, group_names[3]))
        test_5 = read(args, '../Dataset/{}/sparse/{}.txt'.format(args.data_source, group_names[4]))

        test_groups[group_names[0]] = transform(test_1, entity2index)
        test_groups[group_names[1]]  = transform(test_2, entity2index)
        test_groups[group_names[2]]  = transform(test_3, entity2index)
        test_groups[group_names[3]] = transform(test_4, entity2index)
        test_groups[group_names[4]] = transform(test_5, entity2index)
        return test_groups
    else:
        return test_groups # empty dict


# 被 main.py 调用
def load_data(args):
    # 先设置随机数种子
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    train_df = read(args, '../Dataset/' + args.data_source + '/train.txt')

    # 计算总共有多少个 community
    if args.data_source=='Mendeley':
        args.item_num = 103 + 1 # 是 104 不是 103 的原因是因为最后要加一个假想的group用来padding subgraph
        test_df = read(args, '../Dataset/' + args.data_source + '/test.txt')
        valid_df = read(args, '../Dataset/' + args.data_source + '/valid.txt')

    elif args.data_source=='CiteUlike':
        # args.item_num = 584 + 1
        args.item_num = 1214 + 1
        test_df = read(args, '../Dataset/' + args.data_source + '/test.txt')
        valid_df = read(args, '../Dataset/' + args.data_source + '/valid_50x.txt')
    else:
        # Lastfm
        args.item_num = 479 + 1
        test_df = read(args, '../Dataset/' + args.data_source + '/test.txt')
        valid_df = read(args, '../Dataset/' + args.data_source + '/valid.txt')

    print("community(item)的数量：", str(args.item_num))

    # 读entity2id文件, 和 BIQUE/src_data/Mendeley/new_entity2id.txt一样，791992个
    entity2index = read_entity2id('../BiQUE/data/'+args.data_source+'/ent_id')
    entity2index['padding_position'] = len(entity2index)

    # generate subgraph/adj.npy and node_features.npy
    print('开始生成 subgraphs ...')
    download(args, entity2index)
    train_data = transform(train_df, entity2index)
    valid_data = transform(valid_df, entity2index)
    test_data = transform(test_df, entity2index)

    return train_data, valid_data, test_data

def download(args, entity2index):
    if args.data_source == 'Mendeley':
        args.raw_entity_dim = int(args.entity_embed_mendeley.split('_')[4]) * 8
    else:
        args.raw_entity_dim = int(args.entity_embed_citeulike.split('_')[4]) * 8

    if os.path.exists(
            os.path.join('../Dataset/',args.data_source,'subgraph', f'adj_{args.ngh_upper_bound}_{args.raw_entity_dim}_{args.ngh_sampling}.npy')) is True:
        print('subgraph {} already exist!'.format(args.ngh_upper_bound))

    else:
        if args.data_source == 'Mendeley':
            # 调用 generate_entity2context 函数， 返回一个 dict ( ent: ngh)
            entity2context = generate_entity2context(args)
            entity_embs = np.load(args.entity_embed_mendeley)
        elif args.data_source == 'CiteUlike':
            entity2context = gen_ent2con_citeulike(args)
            entity_embs = np.load(args.entity_embed_citeulike)
            entity_embs = np.concatenate((entity_embs, np.zeros((1,args.raw_entity_dim))), axis=0)
        # in Mendeley, args.item_num = 103
        # entity2context 是一个dict
        communities = list(entity2context.keys())[0:args.item_num-1] + ['padding_position']

        # adj 的第一行是中心的社区节点的邻接向量，后面的300行是邻居节点的邻接向量
        adj = np.zeros((args.item_num, args.ngh_upper_bound + 1, args.ngh_upper_bound + 1))
        node_features = np.zeros((args.item_num, args.ngh_upper_bound + 1, args.raw_entity_dim))
        ngh_sampling = json.loads(args.ngh_sampling)
        print('ngh_sampling: {}'.format(ngh_sampling))
        for i in range(args.item_num):
            cmt = communities[i]
            nodes = [cmt]
            # ---------------------------------- 1-hop 一阶邻居节点----------------------------------
            ngh_1st = entity2context[cmt]
            # if len(neighbors) > args.ngh_upper_bound:
            if len(ngh_1st) > ngh_sampling[0]:
                # 判断community的邻居数量是否大于ngh_upper_bound，是的话就sample出ngh_upper_bound个邻居构造子图。
                # nodes = [cmt] + list(random.sample(neighbors, args.ngh_upper_bound))
                nodes += list(random.sample(ngh_1st, ngh_sampling[0]))
            else:
                nodes += sorted(list(ngh_1st))
            # ---------------------------------- 2-hop 二阶邻居节点----------------------------------
            ngh_2nd = []
            for nd in nodes[1:]:
                nds = entity2context[nd]
                if len(nds) > ngh_sampling[1]:
                    ngh_2nd += list(random.sample(nds, ngh_sampling[1]))
                else:
                    ngh_2nd += list(nds)
            nodes += sorted(list(set(ngh_2nd) - set(nodes))) # 注意，二阶邻居不要和一阶邻居以及cmt重复了

            a = lil_matrix((args.ngh_upper_bound + 1, args.ngh_upper_bound + 1))
            for nd in nodes:
                nnghs = entity2context[nd]
                for nngh in nnghs:
                    if nngh in nodes:
                        #print(nodes.index(nd),nodes.index(nngh))
                        a[nodes.index(nd), nodes.index(nngh)] = 1
                        a[nodes.index(nngh), nodes.index(nd)] = 1

            x = np.zeros((args.ngh_upper_bound + 1, args.raw_entity_dim))
            for j in range(len(nodes)):
                x[j] = entity_embs[entity2index[nodes[j]]]
            adj[i] = a.todense()
            node_features[i] = x

        print('adj shape: ',adj.shape)
        print('node_features shape: ', node_features.shape)
        # 保存
        np.save(os.path.join(
            '../Dataset/',args.data_source,'subgraph', f'adj_{args.ngh_upper_bound}_{args.raw_entity_dim}_{args.ngh_sampling}'), adj)
        np.save(os.path.join(
            '../Dataset/',args.data_source,'subgraph', f'node_features_{args.ngh_upper_bound}_{args.raw_entity_dim}_{args.ngh_sampling}'),node_features)
        print('Success generate subgraph of {}'.format(args.data_source))

# 被download调用，用于生成subgraph的embedding和adjacent matrix
def generate_entity2context(args):
    # 读取或生成ent2ngh dataframe, 返回ent_ngh_dict字典
    ent_ngh_dict = {}
    try:
        # 读取ent2ngh dataframe
        ent2ngh = pd.read_csv(os.path.join('../Dataset',args.data_source,'entity2neighbor.csv'),sep='\t',header=None,names=['ent', 'ngh'])
        ent2ngh['ngh'] = [json.loads(ngh.replace('\'', '\"')) for ngh in ent2ngh['ngh']]
        args.max_context_number = 0
        # 前args.item_num个entity是community（item）
        for ngh in ent2ngh['ngh'][:args.item_num-1]:
            if len(ngh) > args.max_context_number:
                args.max_context_number = len(ngh)
        print("max_context_number of the community: ", args.max_context_number)
        # 将 ent2ngh 转化为字典
        for i in range(ent2ngh.shape[0]):
            ent = ent2ngh.loc[i, 'ent']
            ngh = ent2ngh.loc[i, 'ngh']
            ent_ngh_dict[ent] = ngh
    except:
        # 生成ent2ngh dataframe
        print("Preparing entity2neighbor of {}".format(args.data_source))
        entity2id = read_entity2id(os.path.join('../BiQUE/data',args.data_source,'ent_id'))
        kg_data = pd.read_csv(os.path.join('../BiQUE/src_data',args.data_source,'triple.txt'), sep='\t', header=None, names=['h','r','t'])

        ent2ngh = pd.DataFrame(columns=['ent', 'ngh'])
        ent2ngh['ent'] = entity2id.keys()
        ent2ngh['ngh'] = [[] for i in range(len(entity2id))]

        # 注意！！！！这里把ent列设为了index
        ent2ngh = ent2ngh.set_index(['ent'])
        args.max_context_number = 0
        for i in range(kg_data.shape[0]):
            h = kg_data.loc[i, 'h'].strip()
            t = kg_data.loc[i, 't'].strip()
            ent2ngh.loc[h, 'ngh'].append(t)
            ent2ngh.loc[t, 'ngh'].append(h)
            # 前args.item_num个entity是community（item）
            if entity2id[h] < args.item_num-1 and args.max_context_number < len(ent2ngh.loc[h, 'ngh']):
                args.max_context_number = len(ent2ngh.loc[h, 'ngh'])
            elif entity2id[t] < args.item_num-1 and args.max_context_number < len(ent2ngh.loc[t, 'ngh']):
                args.max_context_number = len(ent2ngh.loc[t, 'ngh'])

        print("max_context_number of the community: ", args.max_context_number)
        ent2ngh.to_csv('../Dataset/{}/entity2neighbor.csv'.format(args.data_source),header=None,sep='\t')
        # 返回 entity2context（ent_ngh_dict）
        for ent in ent2ngh.index:
            ngh = ent2ngh.loc[ent,'ngh']
            ent_ngh_dict[ent] = ngh
    ent_ngh_dict['padding_position'] = []
    return ent_ngh_dict

def gen_ent2con_citeulike(args):
    ent_ngh_dict = {}
    try:
        # 读取ent2ngh dataframe
        ent2ngh = pd.read_csv('../Dataset/{}/entity2neighbor.csv'.format(args.data_source), sep='\t',header=None, names=['ent', 'ngh'])
        ent2ngh['ngh'] = [json.loads(ngh for ngh in ent2ngh['ngh'])]
        args.max_context_number = 0
        # 前args.item_num个entity是community（item）
        for ngh in ent2ngh['ngh'][:args.item_num-1]:
            if len(ngh) > args.max_context_number:
                args.max_context_number = len(ngh)

        print("max_context_number of the community: ", args.max_context_number)
        # 将 ent2ngh 转化为字典
        for i in range(ent2ngh.shape[0]):
            ent = ent2ngh.loc[i, 'ent']
        ngh = ent2ngh.loc[i, 'ngh']
        ent_ngh_dict[ent] = ngh
    except:
        args.max_context_number = 0
        entity2id = read_entity2id('../BiQUE/data/{}/ent_id'.format(args.data_source))
        ent_ngh_dict = {k: [] for k in entity2id.keys()}
        kg_data = pd.read_csv('../BiQUE/src_data/{}/triple_1m.txt'.format(args.data_source), sep='\t', header=None,
                              names=['h', 'r', 't'])

        # 添加一步邻居
        for i in range(kg_data.shape[0]):
            h = str(kg_data.loc[i, 'h'])
            t = str(kg_data.loc[i, 't'])
            ent_ngh_dict[h].append(t)
            ent_ngh_dict[t].append(h)
            # 前args.item_num个entity是community（item）

        MAX_SUBGRAPH_NODE = args.ngh_upper_bound
        # 添加二步邻居
        for i in range(len(ent_ngh_dict)):
            k = list(ent_ngh_dict.keys())[i]
            if i >= args.item_num-1:
                break
            # nghs 是一步邻居
            nghs = ent_ngh_dict[k]
            # nghs_add 是二步邻居
            nghs_add = []
            for n in nghs:
                nghs_add += ent_ngh_dict[n]
            # 二步邻居把一步邻居先筛掉
            nghs_add = list(set(nghs_add) - set(nghs))
            if (len(nghs_add) + len(nghs)) > MAX_SUBGRAPH_NODE:
                nghs_add = random.sample(nghs_add, MAX_SUBGRAPH_NODE - len(nghs))
            ent_ngh_dict[k] = list(set(nghs_add + nghs))

            if entity2id[k] < args.item_num -1 and args.max_context_number < len(ent_ngh_dict[h]):
                args.max_context_number = len(ent_ngh_dict[h])

            elif entity2id[t] < args.item_num -1 and args.max_context_number < len(ent_ngh_dict[t]):
                args.max_context_number = len(ent_ngh_dict[t])

        ent2ngh_df = pd.DataFrame(columns=['ent', 'ngh'], dtype=object)
        ent2ngh_df['ent'] = list(ent_ngh_dict.keys())
        ent2ngh_df['ngh'] = list(ent_ngh_dict.values())

        ent2ngh_df.to_csv('../Dataset/{}/entity2neighbor.csv'.format(args.data_source), header=None, index=None, sep='\t')

    ent_ngh_dict['padding_position'] = []
    return ent_ngh_dict


def transform(df, entity2index):
    #把entity的id转化成embedding矩阵下标

    #context_entities = df['center_entities'].map(lambda x: [ [entity2index[j] for j in entity2context[i]] for i in x] ).tolist()
    #context_entities, context_mask = padding(args.entity_number, args.max_context_number, 3, context_entities)
    data = Data(size=df.shape[0], #获取df的行数，shape[1]是列数
                user_entity=np.array(df['user_entity'].map(lambda x: entity2index[x]).tolist()),
                #history_entities = np.array(df['history_entities'].map(lambda x: [entity2index[i] for i in x]).tolist()),
                #candidate_entities=np.array(df['candidate_entities'].map(lambda x: entity2index[x]).tolist())
                center_entities = np.array(df['center_entities'].map(lambda x: [entity2index[i] for i in x]).tolist()),
                history_mask = np.array(df['history_mask'].tolist()),
                history_mask_3d = np.array(df['history_mask_3d'].tolist()),
                #context_entities = np.array(context_entities),
                #context_mask = np.array(context_mask),
                label = np.array(df['label']))
    return data
