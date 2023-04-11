import numpy as np
import tensorflow as tf
import os
from spektral.data import Graph,BatchLoader, DisjointLoader
from spektral.layers import GATConv, GlobalSumPool, GlobalAvgPool, GlobalAttnSumPool, GlobalMaxPool
from tensorflow.keras.regularizers import l2
import math
from transformer_encoder import MultiHeadAttention, point_wise_feed_forward_network
print(tf.__version__)
tf.compat.v1.disable_eager_execution()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #fix the problem from layer normalization
# 测试 tensorflow：tf.test.is_built_with_cuda()，tf.test.is_gpu_available()

class KSGAN(object):
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        self.params = []  # for computing regularization loss
        self._build_inputs(args)
        self._build_layers(args)
        self._build_model(args)
        self._build_train(args)

    def _build_inputs(self, args):
        with tf.name_scope('input'):
            self.user_entity = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name='user_entity')
            self.center_entities = tf.compat.v1.placeholder(
                dtype=tf.int32, shape=[None, args.max_click_history+1], name='center_entities')
            self.history_mask = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None,args.max_click_history], name='history_mask')
            self.history_mask_3d = tf.compat.v1.placeholder(dtype=tf.int32,
                        shape=[None, args.max_click_history, args.max_click_history],name='history_mask_3d')

            self.training = tf.compat.v1.placeholder_with_default(False, shape=(), name='training')
            self.label = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='label')

    def _build_layers(self, args):
        with tf.name_scope('layers'):
            self.dropout = [tf.keras.layers.Dropout(args.dropout_rate) for _ in range(6)]
            self.layernorm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(4)]
            self.gatnorm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(args.prop_time+2)]
            # GAT
            self.convs = [GATConv(channels = int(args.entity_dim / args.attn_heads), activation="elu",
                                kernel_regularizer=l2(args.l2_weight),dropout_rate=args.dropout_rate,
                                attn_heads=args.attn_heads, concat_heads=args.concat_heads, return_attn_coef=True) for _ in range(args.prop_time)]

            # entity embedding 降维
            self.ent_dense = tf.keras.layers.Dense(args.entity_dim)
            # subgraph node embedding 降维
            self.node_dense = tf.keras.layers.Dense(args.entity_dim)
            '''# 最后的dnn的隐藏层
            self.dnn_hidden = tf.keras.layers.Dense(11 * args.entity_dim, activation=tf.nn.leaky_relu)
            # 最后的dnn的输出层
            self.dnn_output = tf.keras.layers.Dense(1)'''
            # feed-forward network
            self.ffn = [point_wise_feed_forward_network(dff=8 * args.entity_dim,d_model=4 * args.entity_dim) for _ in range(2)]
            self.flatten = self.pooling(args.graph_pool)

            # 用于 args.use_context=True 时的context embedding pooling
            self.mean_flatten = self.pooling('GlobalAvgPool')

    def _build_model(self, args):
        with tf.name_scope('Embedding'):
            # (entity_number, entity_dim)
            if args.data_source == 'Mendeley':
                entity_embs = np.load(args.entity_embed_mendeley)[:args.item_num]
            elif args.data_source == 'CiteUlike':
                entity_embs = np.load(args.entity_embed_citeulike)[:args.item_num]
            else:
                entity_embs = np.load(args.entity_embed_citeulike)[:args.item_num]

            self.entity_embeddings = tf.Variable(entity_embs, dtype=tf.float32, name='entity')
            self.entity_embeddings = self.ent_dense(self.entity_embeddings)
            padded_embs = tf.Variable(tf.constant(0, shape=[1, args.entity_dim], dtype=tf.float32),
                                      name='padding_index')
            self.entity_embeddings = tf.concat([self.entity_embeddings, padded_embs], axis=0)
            self.params.append(self.entity_embeddings)
            # (batch_size, (max_click_history + 1), entity_dim)
            center_embs = tf.nn.embedding_lookup(self.entity_embeddings, self.center_entities)
            # (batch_size * (max_click_history + 1), entity_dim)
            self.center_embeddings = tf.reshape(center_embs, shape=[-1, args.entity_dim])

        with tf.name_scope('GraphAttentionNetwork'):
            # 先给出节点的列表，然后再读图
            adj_matrices = np.load(
                os.path.join('../Dataset',args.data_source,f'subgraph/adj_{args.ngh_upper_bound}_{args.raw_entity_dim}_{args.ngh_sampling}.npy'))

            graph_node_embs = np.load(
                os.path.join('../Dataset',args.data_source,f'subgraph/node_features_{args.ngh_upper_bound}_{args.raw_entity_dim}_{args.ngh_sampling}.npy'))
            print('adj shape: {}'.format(adj_matrices.shape))
            print('node_features shape: {}'.format(graph_node_embs.shape))

            self.all_node_embeddings = tf.Variable(graph_node_embs, dtype=tf.float32, name='graph_node_embeddings')
            self.all_node_embeddings = self.node_dense(self.all_node_embeddings)

            # self.all_node_embeddings = self.layer_normalization(self.all_node_embeddings, axis=-1)

            self.params.append(self.all_node_embeddings)

            # (batch_size * (max_click_history + 1), 295, 295)
            center_adjs = tf.reshape(
                tf.nn.embedding_lookup(adj_matrices, self.center_entities), shape=[-1, args.ngh_upper_bound+1, args.ngh_upper_bound+1])
            # (batch_size * (max_click_history + 1), 295, entity_dim)
            ith_node_embs = tf.reshape(
                tf.nn.embedding_lookup(self.all_node_embeddings, self.center_entities),
                                                 shape=[-1, args.ngh_upper_bound + 1, args.entity_dim])
            ith_node_embs = self.gatnorm[0](ith_node_embs)
            if args.gat:
                # historical community 自注意力
                self.self_attn = MultiHeadAttention(d_model=(1+1+args.prop_time) * args.entity_dim, num_heads=args.attn_heads)
                # historical community 和 candidate community 交叉注意力
                self.cross_attn = MultiHeadAttention(d_model=(1+1+args.prop_time) * args.entity_dim, num_heads=args.attn_heads)

                node_embs_iter = [ith_node_embs]
                for i in range(args.prop_time):
                    # ith_node_embs: (batch_size * (max_click_history + 1), 301, entity_dim)
                    # self.gat_coef: (batch_size * (max_click_history + 1), 301, n_heads, 301)
                    ith_node_embs, self.gat_coef = self.convs[i]([ith_node_embs, center_adjs])
                    # self.gat_coef: (batch_size, max_click_history + 1, 301, 301)
                    self.gat_coef = tf.reshape(tf.reduce_mean(self.gat_coef, axis=2), shape=[-1,args.max_click_history+1,args.ngh_upper_bound+1,args.ngh_upper_bound+1])

                    ith_node_embs = self.gatnorm[i+1](ith_node_embs)
                    node_embs_iter.append(ith_node_embs)

                # (batch_size * (max_click_history + 1), 295, (1 + prop_time) * entity_dim) when concat_heads==False
                iterated_node_embs = tf.concat(node_embs_iter, axis=-1)

                iterated_node_embs = self.gatnorm[-1](iterated_node_embs)
                # (batch_size * (max_click_history + 1), (1 + prop_time) * entity_dim)
                subgraph_embeddings = self.flatten(iterated_node_embs)

                # (batch_size * (max_click_history + 1), (1 + 1 + prop_time) * entity_dim)
                self.concat_embeddings = tf.concat([self.center_embeddings, subgraph_embeddings], axis=-1)

                self.full_dim = args.entity_dim + (1 + args.prop_time) * args.entity_dim
            else:
                if args.context:
                    # historical community 自注意力
                    self.self_attn = MultiHeadAttention(d_model=2 * args.entity_dim,
                                                        num_heads=args.attn_heads)
                    # historical community 和 candidate community 交叉注意力
                    self.cross_attn = MultiHeadAttention(d_model=2 * args.entity_dim,
                                                         num_heads=args.attn_heads)

                    # 中心节点embedding拼接avg的邻居节点的embedding
                    # (batch_size * (max_click_history + 1), 295, entity_dim) -> (batch_size * (max_click_history + 1), entity_dim)
                    self.context_embeddings = self.mean_flatten(ith_node_embs)
                    # (batch_size * (max_click_history + 1), entity_dim * 2)
                    self.concat_embeddings = tf.concat([self.center_embeddings, self.context_embeddings], axis=-1)
                    self.full_dim = 2 * args.entity_dim
                else:
                    # historical community 自注意力
                    self.self_attn = MultiHeadAttention(d_model=args.entity_dim,
                                                        num_heads=args.attn_heads)
                    # historical community 和 candidate community 交叉注意力
                    self.cross_attn = MultiHeadAttention(d_model=args.entity_dim,
                                                         num_heads=args.attn_heads)

                    self.concat_embeddings = self.center_embeddings
                    self.full_dim = args.entity_dim

        with tf.name_scope('EmbeddingSlice'):
            # (batch_size * (max_click_history + 1), max_click_history + 1, full_dim)
            # self.concat_embeddings = self.layer_norm(self.concat_embeddings)    # might cause error because of layer_norm, that's why we add "os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'" at beginning
            self.concat_embeddings = self.layer_normalization(self.concat_embeddings, axis=-1)
            # (batch_size, max_click_history + 1, full_dim)
            self.concat_embeddings = tf.reshape(self.concat_embeddings, shape=[-1, (args.max_click_history+1), self.full_dim])
            # (batch_size, max_click_history, full_dim) -> (batch_size, max_click_history, full_dim)
            history_embeddings = tf.reshape(tf.slice(self.concat_embeddings, [0,0,0], [-1,10,-1]), shape=[-1, args.max_click_history, self.full_dim])
            # (batch_size, 1, full_dim) -> (batch_size, full_dim)
            candidate_embeddings = tf.reshape(tf.slice(self.concat_embeddings, [0,10,0], [-1,1,-1]), shape=[-1, self.full_dim])


        with tf.name_scope('Attention_fusion'):

            if args.self_attn:
                # 先扩充 self.history_mask: [batch_size, max_click_history] -> self.history_mask: [batch_size, 1, max_click_history, 1] -> self.history_mask:[batch_size, 1, max_click_history, max_click_history]
                hist_mask_self_attn = tf.expand_dims(self.history_mask_3d, axis=[1])
                # print('hist_mask_self_attn:', hist_mask_self_attn.shape)
                # [batch_size, seq_len_q(max_click_history), seq_len_k(max_click_history)]
                # *************************************** self-attention encoder ******************************************
                # self_attn_output: (batch_size ,max_click_history, full_dim)
                self_attn_output, self.self_attn_weights = self.self_attn(
                    history_embeddings, history_embeddings, history_embeddings, mask=hist_mask_self_attn)
                self_attn_output = self.dropout[0](self_attn_output, training=self.training)
                history_embeddings = self.layernorm[0](self_attn_output)

                '''ffn_output_self = self.ffn[0](out_self)
                ffn_output_self = self.dropout[1](ffn_output_self, training=self.training)
                self_attn_output = self.layernorm[1](out_self + ffn_output_self)'''  # (batch_size, input_seq_len, d_model)
            # *************************************** self-attention encoder end ******************************************

            # *************************************** cross-attention encoder ******************************************
            # (batch_size, 1, full_dim)
            candidate_embeddings = tf.reshape(tf.expand_dims(candidate_embeddings, 1), [-1, 1, self.full_dim])

            if args.cross_attn:
                # [batch_size, seq_len_q (1), seq_len_k(max_click_history)]
                hist_mask_cross_attn = tf.expand_dims(tf.expand_dims(self.history_mask, 1), 1)
                # q: candidate_embeddings, k: self_attn_output, v: self_attn_output
                # cross_attn_output : (batch_size , 1, full_dim)
                cross_attn_output, self.cross_attn_weights = self.cross_attn(
                    candidate_embeddings, history_embeddings, history_embeddings, mask=hist_mask_cross_attn)
                cross_attn_output = self.dropout[2](cross_attn_output, training=self.training)
                history_embeddings = self.layernorm[2](cross_attn_output)
                self.cross_attn_weights = tf.reshape(self.cross_attn_weights, shape=[-1, args.max_click_history])

                '''ffn_output_cross = self.ffn[1](out_cross)
                ffn_output_cross = self.dropout[3](ffn_output_cross, training=self.training)
                history_embeddings = self.layernorm[3](out_cross + ffn_output_cross)'''
                # *************************************** cross-attention encoder end ******************************************
                # history_embeddings, self.cross_attn_weights = self._attention(history_embeddings, self.history_mask, candidate_embeddings, args)
            else:
                hist_mask = tf.cast(tf.expand_dims(self.history_mask, 1)/args.max_click_history, tf.float32)
                # (batch_size, 1, 10) * (batch_size, 10, full_dim)
                history_embeddings = tf.matmul(hist_mask, history_embeddings)
                history_embeddings = tf.reshape(history_embeddings,shape=[-1, 1, self.full_dim])
                #history_embeddings = tf.reduce_mean(history_embeddings, axis=1)

        # DNN
        #不用user_embeddings
        with tf.name_scope('Prediction'):
            '''output_embedding = tf.concat([history_embeddings, candidate_embeddings], axis=-1)
            output_dropout = self.dropout[4](output_embedding,training=self.training)

            hidden_layer= self.dnn_hidden(output_dropout)
            # hidden_layer = tf.compat.v1.layers.dense(output_dropout, units=11 * self.full_dim, activation=tf.nn.leaky_relu, name='hidden')
            hidden_drop = self.dropout[5](hidden_layer, training=self.training)

            self.scores_unnormalized = self.dnn_output(hidden_drop)
            # self.scores_unnormalized = tf.compat.v1.layers.dense(hidden_drop, units=1, activation=tf.nn.leaky_relu,name='dnn')
            self.scores_unnormalized = tf.reshape(self.scores_unnormalized, shape=[-1])'''

            # dot product
            self.scores_unnormalized = tf.matmul(history_embeddings, candidate_embeddings, transpose_b=True)
            self.scores_unnormalized = tf.reshape(self.scores_unnormalized, shape=[-1])
            self.scores = tf.sigmoid(self.scores_unnormalized)

    def _build_train(self, args):
        with tf.name_scope('train'):
            self.base_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.scores_unnormalized))
            self.l2_loss = tf.Variable(tf.constant(0., dtype=tf.float32), trainable=False)
            for param in self.params:
                self.l2_loss = tf.add(self.l2_loss, args.l2_weight * tf.nn.l2_loss(param))

            self.loss = self.base_loss + self.l2_loss
            self.optimizer = tf.compat.v1.train.AdamOptimizer(args.lr).minimize(self.loss)

    def pooling(self, pooling_type):
        if pooling_type == 'GlobalSumPool':
            return GlobalSumPool()
        elif pooling_type == 'GlobalAvgPool':
            return GlobalAvgPool()
        elif pooling_type == 'GlobalAttnSumPool':
            return GlobalAttnSumPool()
        elif pooling_type == 'GlobalMaxPool':
            return GlobalMaxPool()
        else:
            print('pool error!')

    def layer_normalization(self, emb, axis):
        # For each sample x_i in inputs with k features
        # x_i_normalized = (x_i - mean_i) / sqrt(var_i + epsilon)
        # epsilon is small float to avoid dividing by zero. Defaults to 1e-3
        mean, var = tf.nn.moments(emb, axes=axis)
        mean = tf.expand_dims(mean, axis=axis)
        var = tf.expand_dims(var, axis=axis)
        eps = tf.constant(1e-5, dtype=tf.float32, name='epsilon')
        norm_emb = (emb - mean) / tf.pow(var + eps, 0.5)
        return norm_emb

    def train(self, sess, feed_dict):
        return sess.run([self.loss, self.optimizer], feed_dict)

    def eval(self, sess, feed_dict):
        # input is of test size
        user, label, scores = sess.run([self.user_entity, self.label, self.scores], feed_dict)
        # user, label, scores, self_weights, cross_weights= sess.run([self.user_entity, self.label, self.scores, self.self_attn_weights, self.cross_attn_weights,], feed_dict)
        # return list(user), list(label), list(scores), self_weights, cross_weights
        return list(user), list(label), list(scores)