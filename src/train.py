from ksgan import KSGAN
import tensorflow as tf
import numpy as np
from evaluate import score
import json
import os
from time import time

# tf.compat.v1.disable_v2_behavior()
def get_feed_dict(model, data, training, start, end):
    feed_dict = {model.user_entity: data.user_entity[start:end],
                 model.center_entities: data.center_entities[start:end],
                 model.history_mask: data.history_mask[start:end],
                 model.history_mask_3d: data.history_mask_3d[start:end],
                 model.training: training,
                 model.label: data.label[start:end]}
    return feed_dict

def eval_model(args, sess, model, valid_data, save_path, is_test):
    cur_stage = 'test' if is_test else 'valid'
    valid_start_list = list(range(0, valid_data.size, args.batch_size))
    user, label, scores = [], [], []
    if is_test:
        # self_attn_weights = np.zeros(shape=(test_data.size, args.max_click_history, args.max_click_history))
        # cross_attn_weights = np.zeros(shape=(test_data.size, args.max_click_history))
        # gat的权重 (74609, 301, 4, 301) 太大了，分配不了那么多
        # gat_coef = np.zeros(shape=(args.batch_size, args.ngh_upper_bound+1, args.attn_heads, args.ngh_upper_bound+1))
        pass

    start_t = time()
    for start in valid_start_list:
        end = start + args.batch_size
        if is_test:
            # u, l, s, _, _ = model.eval(sess, get_feed_dict(model, valid_data, False, start, end))
            u, l, s = model.eval(sess, get_feed_dict(model, valid_data, False, start, end))
            # gat_coef = gat
            # self_attn_weights[start:end] = self_weights
            # cross_attn_weights[start:end] = cross_weights
        else:
            u, l, s = model.eval(sess, get_feed_dict(model, valid_data, False, start, end))
        user += u
        label += l
        scores += s
    end_t = time()

    auc, ndcgs, precisions, hit_ratio = score(user, label, scores, json.loads(args.top_k))
    output_dict = {
        '{}_time'.format(cur_stage): end_t - start_t,
        # 'loss': float(format(losses, '.4f')),
        'auc_score': auc,
        'ndcgs': ndcgs,
        'hit_ratio': hit_ratio,
        'precision': precisions,
    }

    if is_test:
        # np.savetxt(os.path.join(save_path, 'cross_attn_weights'), cross_attn_weights)
        # 三维及以上的数组要用savez存储
        # np.savez(os.path.join(save_path, 'self_attn_weights'), self_attn_weights)
        # np.savez(os.path.join(save_path, 'gat_coef'), gat_coef)
        # np.savetxt(os.path.join(save_path, 'score.txt'), scores)
        pass

    return output_dict

def train(args, train_data, valid_data, test_data, sparse_tests=None):
    # 设置 tensorflow 的随机数种子
    tf.random.set_seed(args.seed)
    # 激活 Tensorflow 确定性功能
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    model = KSGAN(args)

    # 生成模型的文件夹以及config.json
    assert args.save_path
    save_suffix = f"{args.model}_{args.data_source[0]}_{args.ngh_sampling}_{args.lr}_{args.batch_size}_{args.l2_weight}_{args.dropout_rate}_{args.ngh_upper_bound}_{args.attn_heads}_{args.prop_time}"
    save_path = os.path.join(args.save_path, save_suffix)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        best_ndcg10 = 0.0
        best_epoch = 0

        for step in range(args.n_epochs):
            # training
            train_start_list = list(range(0, train_data.size, args.batch_size))
            np.random.shuffle(train_start_list)
            losses = 0
            for start in train_start_list:
                end = start + args.batch_size
                loss, optimizer = model.train(sess, get_feed_dict(model, train_data, True, start, end))
                losses += loss
            print(f'Epochs {step}: loss {losses}')

            # valid
            if step % args.valid_epoch==0:
                print('*' * 60 + ' valid stage ' + '*' * 60)
                output_dict = eval_model(args, sess, model, valid_data, save_path, is_test=False)

                with open(os.path.join(save_path, 'train.log'), 'a+', encoding='utf-8') as log_file:
                    log_file.write("Epoch: {}\n".format(step))
                    log_file.write("\t Valid: {}\n".format(output_dict))

                print(json.dumps(output_dict))
                if output_dict['ndcgs'][2] >= best_ndcg10:
                    best_ndcg10 = output_dict['ndcgs'][2]
                    best_epoch = step
                # -----------------------------------------   test stage    -----------------------------------------
                if args.data_source=='CiteUlike':
                    if step <= args.from_epoch_test:
                        continue
                print('*' * 60 + ' test stage ' + '*' * 60)
                output_dict = eval_model(args, sess, model, test_data, save_path, is_test=True)
                print(json.dumps(output_dict))
                with open(os.path.join(save_path, 'train.log'), 'a+', encoding='utf-8') as log_file:
                    log_file.write("\t TEST: {}\n".format(output_dict))

                # ----------------------------------------- test for sparse data -----------------------------------------
                if args.sparse_test:
                    print('*' * 60 + ' sparse test ' + '*' * 60)
                    # sparse_tests 是个字典
                    for sparse_level in sparse_tests:
                        # sparse_level: user historical interactions <2, <3, <5 and <11
                        test_group = sparse_tests[sparse_level]
                        output_dict = eval_model(args, sess, model, test_group, save_path, is_test=True)
                        print('{}: {}'.format(sparse_level, json.dumps(output_dict)))
                        with open(os.path.join(save_path, 'train.log'), 'a+', encoding='utf-8') as log_file:
                            log_file.write("\t {}: {}\n".format(sparse_level, output_dict))

        print(f'best valid epoch: {best_epoch}, best ndcg10: {best_ndcg10}')
        with open(os.path.join(save_path, 'train.log'), 'a+', encoding='utf-8') as log_file:
            log_file.write("Best Epoch: {}\n".format(best_epoch))
            log_file.write("\t Best Test Ndcg@10: {}\n".format(best_ndcg10))
