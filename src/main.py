import argparse
from data_loader import load_data, load_sparse_test
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,default='KSGAN',help='model name for ablation study')
parser.add_argument('--data_source',type=str,default='Mendeley',help='name of used dataset')
parser.add_argument('--entity_embed_mendeley',type=str,default='../BiQUE/logs/BiQUE_Mendeley_wN3_800_48_0.15_0.1_0/entity_embedding.npy',help='path of mendeley entity embedding')
parser.add_argument('--entity_embed_citeulike',type=str,default='../BiQUE/logs/BiQUE_CiteUlike_wN3_2000_32_0.01_0.1_0/entity_embedding.npy',help='path of lastfm entity embedding')

#parser.add_argument('--entity_embed_lastfm',type=str,default='../BiQUE/logs/BiQUE_Lastfm_wN3_3000_128_0.15_0.1_0/entity_embedding.npy',help='path of lastfm entity embedding')
parser.add_argument('--gpu_id',type=str,default='1',help='which gpu to choose to run')
parser.add_argument('--seed',type=int,default = 2021, help='the random seed for numpy and tensorflow')

parser.add_argument('--save_path', type=str, default='logs/')
parser.add_argument('--valid_epoch', default=3, type=float, help="Number of epochs before valid.")
parser.add_argument('--ngh_sampling', type=str, default='[25,10]', help='node-wise neighbor sampling of subgraph')


parser.add_argument('-context', action='store_true',help='whether to use context embeddings')
parser.add_argument('-gat', action='store_true', help='whether use a GAT network')
parser.add_argument('-self_attn', action='store_true', help='whether use a self-attention network')
parser.add_argument('-cross_attn', action='store_true', help='whether use a cross-attention network')


parser.add_argument('--graph_pool', type=str, default='GlobalAttnSumPool', help='the type of pool used after GAT')

parser.add_argument('--concat_heads', type=bool, default=True,
                    help='whether to concatenate attention heads in GAT. If flase, then average the heads')
parser.add_argument('--load_model', type=str, default='',
                    help='whether to load model')

parser.add_argument('--prop_time', type=int, default=2,
                    help='whether to concatenate attention heads in GAT. If flase, then average the heads')
# parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for optimizer')

parser.add_argument('--max_click_history', type=int, default=10, help='number of sampled click history for each user')
parser.add_argument('--ngh_upper_bound', type=int, default=300, help='max number of neighbors for each center node')
parser.add_argument('--l2_weight', type=float, default=0.01, help='weight of l2 regularization')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='the weight of dropout in dnn')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='number of samples in one batch')
parser.add_argument('--from_epoch_test', default=30, type=float, help="from_epoch_test")

parser.add_argument('--n_epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--attn_heads', type=int, default=4, help='number of attention heads in GAT')
parser.add_argument('--KGE', type=str, default='BiQUE', help='knowledge graph embedding method, please ensure that the specified input file exists')
parser.add_argument('--entity_dim', type=int, default=64,
                    help='dimension of entity embeddings, please ensure that the specified input file exists')
parser.add_argument('--top_k', type=str, default='[1,5,10,15,20,25,30,40,50]', help='top k')
parser.add_argument('--sparse_test', type=bool, default=False, help='sparse test experiments: using different sparse groups of interactions')

args = parser.parse_args()

print(args)

train_data, valid_data, test_data = load_data(args)
if args.data_source=='CiteUlike':
    args.sparse_test = False

if args.sparse_test:
    # sparse_test_groups: {sparse_level: test_data}
    sparse_test_groups = load_sparse_test(args)
    train(args, train_data, valid_data, test_data, sparse_tests = sparse_test_groups)
else:
    train(args, train_data, valid_data, test_data, sparse_tests = {})


