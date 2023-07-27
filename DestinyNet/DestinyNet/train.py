import scanpy as sc
import matplotlib.pyplot as plt
import torch
from torch.nn import DataParallel
import os
import anndata as ad
from sklearn import preprocessing
from .utils import get_args
import cospar as cs
import scanpy as sc
import numpy as np
import operator
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.preprocessing import MaxAbsScaler
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from math import sqrt
from random import choice
from torch.utils.data  import Dataset, DataLoader
import pandas as pd
from math import sqrt
import torch
from tqdm import trange
import torch.nn as nn
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        #print('channel:',c)
        #print('size:',x.shape)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        #print('y_size:',y.shape)
        return x * y.expand_as(x)
class MutiheadAttention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v,num_heads):
        super(MutiheadAttention, self).__init__()
        self.dim_q = dim_k # 一般默认 Q=K
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_units=dim_k
        self.num_heads=num_heads
        #定义线性变换函数
        self.linear_q = nn.Linear(input_dim, dim_k, bias=False)
        self.linear_k = nn.Linear(input_dim, dim_k, bias=False)
        self.linear_v = nn.Linear(input_dim, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # x: batch_size, seq_len, input_dim
        q = self.linear_q(x)  # batch_size, seq_len, dim_k
        k = self.linear_k(x)  # batch_size, seq_len, dim_k
        v = self.linear_v(x)  # batch_size, seq_len, dim_v
        split_size = self.num_units // self.num_heads
        q = torch.stack(torch.split(q, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        k = torch.stack(torch.split(k, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        v = torch.stack(torch.split(v, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        scores = torch.matmul(q, k.transpose(2, 3))
        scores = scores / (self.dim_k ** 0.5)
        
        scores = F.softmax(scores, dim=3)
        ## out = score * V
        out = torch.matmul(scores, v)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0) 

        return out

class ResidualBlock(torch.nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv1d(channels,channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(channels,channels,kernel_size=3,padding=1)
        self.se=SELayer(channels,16)
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        y=self.se(y)
        return F.relu(x+y)

class DestinyNet(nn.Module):
    def __init__(self, len_embedding, num_relations):
        nn.Module.__init__(self)
        self.att=MutiheadAttention(len_embedding*2,512,512,64)
        self.layernorm=nn.LayerNorm(512)
        self.conv1 = nn.Conv1d(1, 32, 4)  # 输入通道数为1，输出通道数为6
        self.relu1=nn.LeakyReLU(0.2, inplace=True)
        self.rblock1 = ResidualBlock(32)
        self.conv2 = nn.Conv1d(32,64, 4)  # 输入通道数为6，输出通道数为16
        self.batchn1=nn.BatchNorm1d(64)
        self.relu2= nn.LeakyReLU(0.2, inplace=True)
        self.rblock2 = ResidualBlock(64)
        self.conv3=nn.Conv1d(64,128,4)
        self.batchn2=nn.BatchNorm1d(128)
        self.relu3= nn.LeakyReLU(0.2, inplace=True)
        self.rblock3 = ResidualBlock(128)
        self.conv4=nn.Conv1d(128,256,4)
        self.batchn3=nn.BatchNorm1d(256)
        self.relu4= nn.LeakyReLU(0.2, inplace=True)
        self.rblock4 = ResidualBlock(256)
        self.dropout=nn.Dropout()
        self.fc1 = nn.Linear(7424, num_relations)
        
    def forward(self, x):
        # 输入x -> conv1 -> relu -> 2x2窗口的最大池化
        x=self.att(x)+x
        x=self.layernorm(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.max_pool1d(x, 2)
        x=self.rblock1(x)
        # # 输入x -> conv2 -> relu -> 2x2窗口的最大池化
        x = self.conv2(x)
        x=self.batchn1(x)
        x=self.relu2(x)
        x = F.max_pool1d(x, 2)
        x=self.rblock2(x)
        
        x = self.conv3(x)
        x=self.batchn2(x)
        x=self.relu3(x)
        x = F.max_pool1d(x, 2)
        x=self.rblock3(x)
        
        x = self.conv4(x)
        x=self.batchn3(x)
        x=self.relu4(x)
        x = F.max_pool1d(x, 2)
        x=self.rblock4(x)
        # # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        x = x.view(x.size()[0], -1)
        x=self.dropout(x)
        x=self.fc1(x)
        return x
class TrainDataset(Dataset):
    # 构造器初始化方法
    def __init__(self,length,traincell1,traincell2,train_rel,adata_orig):
        self.length=length
        self.traincell1=traincell1
        self.traincell2=traincell2
        self.adata_orig=adata_orig
        self.train_rel=train_rel
    # 重写getitem方法用于通过idx获取数据内容
    def __getitem__(self, idx):
        cell1_id=int(self.traincell1[idx])
        cell2_id=int(self.traincell2[idx])
        gene1=self.adata_orig.X[cell1_id].toarray()
        gene1=torch.tensor(gene1)
        gene2=self.adata_orig.X[cell2_id].toarray()
        gene2=torch.tensor(gene2)       
        genetype=self.train_rel[idx]
        genetype=torch.tensor(genetype)
        return gene1,gene2,genetype

    # 重写len方法获取数据集大小
    def __len__(self):
        return self.length



def train(args):
    device=args.device
    adata_orig = cs.hf.read(args.adata_orig)
    traincell1 = np.loadtxt(args.traincell1)
    traincell2 = np.loadtxt(args.traincell2)
    testcell1 = np.loadtxt(args.testcell1)
    testcell2 = np.loadtxt(args.testcell2)
    train_rel = np.loadtxt(args.train_rel, dtype=int)
    test_rel = np.loadtxt(args.test_rel, dtype=int)
    traindataset = TrainDataset(len(train_rel),traincell1,traincell2,train_rel,adata_orig)
    if(args.type_of_geneEnc==0 and args.Dropout_for_geneEnc==False):
        geneEnc=nn.Sequential(
            nn.Linear(args.len_geneExp, args.len_embedding)
                         )
    elif(args.type_of_geneEnc==0 and args.Dropout_for_geneEnc==True):
        geneEnc=nn.Sequential(
            nn.Dropout(),
            nn.Linear(args.len_geneExp, args.len_embedding)
                         )
    elif(args.type_of_geneEnc==1 and args.Dropout_for_geneEnc==False):  
        geneEnc=nn.Sequential(
        nn.Linear(args.len_geneExp, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
         nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, args.len_embedding),)
    
    else:
        geneEnc=nn.Sequential(
        nn.Dropout(),
        nn.Linear(args.len_geneExp, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
         nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, args.len_embedding),)
        
    geneDec = nn.Sequential(
        nn.Linear(args.len_embedding, 512),
        nn.Linear(512, args.len_geneExp),
    )    
    genemap = nn.Sequential(
         #nn.Dropout(),
         nn.Linear(args.len_embedding, 100),
         nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.Linear(100, 100),
       nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.Linear(100, 100),
       nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.Linear(100, args.len_embedding),
)
    model = DestinyNet(args.len_embedding, args.num_relations)
    geneEnc=geneEnc.to(device)
    geneDec=geneDec.to(device)
    model=model.to(device)
    genemap=genemap.to(device)
    traindataloader = DataLoader(traindataset, batch_size=args.batch_size, num_workers=16, shuffle=True, drop_last=False,pin_memory = True)
    optimizer = torch.optim.Adam(
    [{'params': geneEnc.parameters()},
     {'params': geneDec.parameters()},
    {'params': model.parameters()},
     {'params': genemap.parameters()},
        ],
    lr=args.learning_rate,weight_decay=args.weight_decay)
    mse = nn.MSELoss(reduce=False)
    criterion_rec = nn.MSELoss()
    print(geneEnc)
    for epoch in range(args.epochs):
        for i, onebatch in enumerate(traindataloader):
       
            geneExp1, geneExp2, labels = onebatch[0], onebatch[1], onebatch[2]
            geneExp1=torch.reshape(geneExp1,(geneExp1.shape[0],geneExp1.shape[2]))
            geneExp2=torch.reshape(geneExp2,(geneExp2.shape[0],geneExp2.shape[2]))
            geneExp1, geneExp2, labels =geneExp1.to(device),geneExp2.to(device),labels.to(device)
            geneEmb1, geneEmb2 = geneEnc(geneExp1), geneEnc(geneExp2)
            geneExp_rec1, geneExp_rec2 = geneDec(geneEmb1), geneDec(geneEmb2)
           
        
            item2_zero = torch.where(labels!=args.num_relations)
            geneExp3_zero = geneExp1[item2_zero]
            geneExp4_zero = geneExp2[item2_zero]
            geneEmb3_zero, geneEmb4_zero = geneEnc(geneExp3_zero), geneEnc(geneExp4_zero)
            mapgene_zero = genemap(geneEmb3_zero)
            #mapgene_one = genemap(geneEmb3_one)
            
            loss4 = torch.mean(torch.sum(mse(mapgene_zero, geneEmb4_zero), dim=1))
            #loss5 = torch.mean(torch.sum(mse(mapgene_one, geneEmb4_one), dim=1))

            loss1 = criterion_rec(geneExp1,geneExp_rec1)
            loss2 = criterion_rec(geneExp2,geneExp_rec2)
            geneEmbs = torch.cat((geneEmb1, geneEmb2), 1)
            geneEmbs=torch.reshape(geneEmbs,(geneEmbs.shape[0],1,geneEmbs.shape[1]))

            hang=onebatch[0].shape[0]
            outputs = model(geneEmbs)
            
            criterion=nn.CrossEntropyLoss()
            loss3 = criterion(outputs, labels)

            print("loss1:",loss1)
            print("loss2:",loss2)
            print("loss3:",loss3)
            print("loss4:", loss4)

            loss=1*(loss1+loss2)+1000*loss3+10*loss4
            print('epoch=' + str(epoch) + ', batch=' + str(i) + ' ' + ' loss=', loss)

            optimizer.zero_grad()  # 清除梯度
            loss.backward()
            optimizer.step()
          
        print('loss after epoch' + str(epoch) + ': ', loss)
    state = {'geneEnc': geneEnc.state_dict(),
    'geneDec':geneDec.state_dict(),
             'model': model.state_dict(),
                      'genemap':genemap.state_dict()
    }
    torch.save(state, args.save_path)
    ##GetEmbedding
    geneEnc.eval()
    genemap.eval()
    mapembedding = []
    all_embeddings = []
    with torch.no_grad():
        for i in trange(adata_orig.shape[0]):
            x = adata_orig.X[i].toarray()
            geneExp = torch.tensor(x).to(device)
            #geneExp = torch.reshape(geneExp, (1, x.shape[0]))
            geneExp = geneExp.to(torch.float32)
            geneEmbedding = geneEnc(geneExp)
            mapEmbedding = genemap(geneEmbedding)
            mapEmbedding = mapEmbedding.cpu().detach().numpy()
            geneEmbedding = geneEmbedding.cpu().detach().numpy()
            all_embeddings.append(geneEmbedding)
            mapembedding.append(mapEmbedding)
    all_embeddings = np.array(all_embeddings)
    mapembedding = np.array(mapembedding)
    np.savez(args.embeddings_path, all_embeddings)
    np.savez(args.mapping_path, mapembedding)
if __name__ == "__main__":
    args = get_args()
    train(args)