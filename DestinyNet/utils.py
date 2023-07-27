import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--len_geneExp', type=int, default=17695, help='adata.X size')
    parser.add_argument('--len_embedding', type=int, default=256, help='embedding size')
    parser.add_argument('--save_path', type=str, default='/home/zhengtuo/songtao/pei.pth', help='model save path')
    parser.add_argument('--adata_orig', type=str, default="/home/zhengtuo/songtao/pei_17695.h5ad", help='Path to adata_orig file')
    parser.add_argument('--traincell1', type=str, default='/home/zhengtuo/songtao/3traincell1.txt', help='Path to traincell1 file')
    parser.add_argument('--traincell2', type=str, default='/home/zhengtuo/songtao/3traincell2.txt', help='Path to traincell2 file')
    parser.add_argument('--testcell1', type=str, default='/home/zhengtuo/songtao/3testcell1.txt', help='Path to testcell1 file')
    parser.add_argument('--testcell2', type=str, default='/home/zhengtuo/songtao/3testcell2.txt', help='Path to testcell2 file')
    parser.add_argument('--train_rel', type=str, default='/home/zhengtuo/songtao/3trainrel.txt', help='Path to train_rel file')
    parser.add_argument('--test_rel', type=str, default='/home/zhengtuo/songtao/3testrel.txt', help='Path to test_rel file')
    parser.add_argument('--Dropout_for_geneEnc', type=bool, default=True, help='use Dropout in geneEnc')
    parser.add_argument('--type_of_geneEnc', type=int, default=1, help='1 for multi layer MLP,0 for single layer MLP(if the data size is few)')
    parser.add_argument('--num_relations', type=int, default=3, help='number of relations')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay for the optimizer')
    
    parser.add_argument('--device', type=str, default='cuda:0', help='id of gpu')
    parser.add_argument('--embeddings_path', type=str, default='/home/zhengtuo/songtao/embedding.txt', help='embedding save path')
    parser.add_argument('--mapping_path', type=str, default='/home/zhengtuo/songtao/mapping.txt', help='mapping save path')
        
        
    args = parser.parse_args()
    return args
