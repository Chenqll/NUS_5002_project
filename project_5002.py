# %%
import pandas as pd
# conda install -n base ipykernel --update-deps --force-reinstall

# %%
data=pd.read_csv('data/videos.csv',encoding='utf-8',encoding_errors='ignore')
data.head()
data.dropna(axis=0, how='any', inplace=True)


# %%
data_SA=data.loc[:,['comments Sentiment','Comment Detail']]
data_SA

# %%
# 划分数据集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_SA['Comment Detail'].values, # ndarray
                                                    data_SA['comments Sentiment'].values,
                                                    train_size=0.7,
                                                    random_state=100)



# %%
from torch.utils.data import Dataset, DataLoader
import torch

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)


# %%
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# next(iter(train_loader))
for i, batch in enumerate(train_loader):
    print(batch)
    break


# %%
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())

print(device)

from torch import nn
from transformers import BertModel, BertTokenizer
from transformers import AdamW
from tqdm import tqdm

num_class=3

class BertClassificationModel(nn.Module):
    def __init__(self,hidden_size=768): # bert默认最后输出维度为768
        super(BertClassificationModel, self).__init__()
        model_name = 'bert-base-cased'
        # 读取分词器
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        # 读取预训练模型
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=model_name)

        for p in self.bert.parameters(): # 冻结bert参数
                p.requires_grad = False
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, batch_sentences): # [batch_size,1]
        # 编码
        sentences_tokenizer = self.tokenizer(batch_sentences,
                                             truncation=True,
                                             padding=True,
                                             max_length=512,
                                             add_special_tokens=True)
        input_ids=torch.tensor(sentences_tokenizer['input_ids']).to(device) # 变量
        attention_mask=torch.tensor(sentences_tokenizer['attention_mask']).to(device) # 变量
        bert_out=self.bert(input_ids=input_ids,attention_mask=attention_mask) # 模型

        last_hidden_state =bert_out[0].to(device) # [batch_size, sequence_length, hidden_size] # 变量
        bert_cls_hidden_state=last_hidden_state[:,0,:].to(device) # 变量
        fc_out=self.fc(bert_cls_hidden_state) # 模型
        return fc_out

model=BertClassificationModel()
model=model.to(device)
optimizer=AdamW(model.parameters(),lr=1e-4)
loss_func=nn.CrossEntropyLoss()
loss_func=loss_func.to(device)


# %%

def train():
    model.train()
    for i,(data,labels) in enumerate(tqdm(train_loader)):

        out=model(data) # [batch_size,num_class]
        loss=loss_func(out.cpu(),labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i%5==0:
            out=out.argmax(dim=-1)
            acc=(out.cpu()==labels).sum().item()/len(labels)
            print(i, loss.item(), acc) # 一个batch的数据

for epoch in range(500):
    train()



def test():
    model.eval()
    correct = 0
    total = 0
    for i,(data,labels) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            out=model(data) # [batch_size,num_class]

        out = out.argmax(dim=1)
        correct += (out.cpu() == labels).sum().item()
        total += len(labels)

    print(correct / total)
    # 0.7813171080887616

test()
