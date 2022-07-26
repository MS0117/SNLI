import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import RobertaModel, RobertaTokenizer,RobertaConfig,RobertaForSequenceClassification, DataCollatorWithPadding, AdamW,get_scheduler
from datasets import load_dataset,load_metric
from torch.utils.data import DataLoader

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

checkpoint="roberta-base"
tokenizer=RobertaTokenizer.from_pretrained(checkpoint)

model=RobertaForSequenceClassification.from_pretrained(checkpoint,num_labels=3)

model2=RobertaModel.from_pretrained(checkpoint)

class multiclasshead(nn.Module):
  def __init__(self,num_dimension,num_class,dropout):
    super(multiclasshead,self).__init__()
    self.linear=nn.Linear(num_dimension,num_dimension)
    self.relu=F.ReLu
    self.dropout=nn.dropout(dropout)
    self.classify=nn.Linear(num_dimension,num_class)

  def forward(self,x):
    x=x[:,0,:]
    x=self.linear(x)
    x=self.relu(x)
    x=self.dropout(x)
    y=self.classify(x)

    return y

raw_dataset=load_dataset("snli")

print(raw_dataset)

np.unique(raw_dataset['train'][:]['label'])

(torch.Tensor(raw_dataset['train'][:]['label'])==0).sum()

new_dataset=raw_dataset.filter(lambda x: x['label']== 0 or x['label']==1 or x['label']==2)

#(torch.Tensor(new_dataset['train'][:]['label'])==0).sum()

#(torch.Tensor(raw_dataset['train'][:]['label'])==1).sum()

#(torch.Tensor(raw_dataset['train'][:]['label'])==2).sum()

#(torch.Tensor(raw_dataset['train'][:]['label'])==-1).sum()


def tokenizing_function(data):
  return tokenizer(data['premise'],data['hypothesis'],truncation=True)


tokenized_dataset=new_dataset.map(tokenizing_function,batched=True)

print(tokenized_dataset)

tokenized_dataset['train'][:5]

tokenized_dataset=tokenized_dataset.remove_columns(['premise','hypothesis'])
tokenized_dataset=tokenized_dataset.rename_column('label','labels')
tokenized_dataset.set_format('torch')

tokenized_dataset['train']

(tokenized_dataset['train'][:]['labels']==-1).sum()

data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
#train_datasets_sampled =  tokenized_dataset["train"].shuffle(seed=42).select(range(100))      #train_datasets_sampled =  tokenized_dataset["train"].shuffle(seed=42).select(range(2000))
#testdatasets_sampled =  tokenized_dataset["test"].shuffle(seed=42).select(range(100))      #train_datasets_sampled =  tokenized_dataset["train"].shuffle(seed=42).select(range(2000))
train_dataloader=DataLoader(tokenized_dataset['train'],batch_size=16,shuffle=True,collate_fn=data_collator)
test_dataloader=DataLoader(tokenized_dataset['test'],batch_size=16,collate_fn=data_collator)
validation_dataloader=DataLoader(tokenized_dataset['validation'],batch_size=16,collate_fn=data_collator)

for batch in train_dataloader:
  break
{k:v.shape for k,v in batch.items()}

output=model(**batch)
print(output.loss)

optimizer=AdamW(model.parameters(),lr=3e-3)
num_epoch=3
num_training_steps=len(train_dataloader)*num_epoch

scheduler=get_scheduler("linear", num_training_steps=num_training_steps, optimizer=optimizer,num_warmup_steps=0)



device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model.train()

progress_bar=tqdm(range(num_training_steps))
for i in range(num_epoch):
  for batch in train_dataloader:
    inputs={k:v.to(device) for k,v in batch.items()}
    #print(inputs)
    outputs=model(**inputs)
    loss=outputs.loss
    loss.backward()
    optimizer.step()

    scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)

model.eval()
cnt=0
total=0
for batch in test_dataloader:
  inputs={k:v.to(device) for k,v in batch.items()}
  outputs=model(**inputs)

  with torch.no_grad():
    outputs=model(**inputs)
    logits=outputs.logits
    predict=torch.argmax(logits,dim=-1).to(device)
    total+=predict.size(0)
    cnt+=(predict==batch["labels"].to(device)).sum()
    print("cnt",cnt)

cnt=cnt.float()    
Accuracy=cnt/total
print("ACC",Accuracy)
print("total",total)
print("cnt",cnt)
