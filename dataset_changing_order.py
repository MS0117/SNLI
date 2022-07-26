
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
  def __init__(self,max_seq_len,tokenizer,file_dir):
    premise=[]
    hypothesis=[]
    labels=[]
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    with open(file_dir, 'r') as f:
      while True:
        src=f.readline()
        if src=='':
          break
        #print(src)
        src=src.split('\t')
        #print(src)
        premise.append(src[0])
        try:
          hypothesis.append(src[1])
        except:
          print(src)
        labels.append(src[2].strip('\n'))
    labels=[label_set[label] for label in labels]
    self.len=len(labels)
    self.premise=premise
    self.hypothesis=hypothesis
    self.labels=labels
    self.max_seq_len=max_seq_len
    self.tokenizer=tokenizer
    #print("premise",self.premise)

    #print("hypothesis",self.hypothesis)
    #print("labels",self.labels)

  def __getitem__(self,index):

      def tokenizing_function(data):
          return self.tokenizer(data["premise"], data["hypothesis"])

      #print("tokenized_premise",self.tokenizer(self.premise[index]))
      input = {}

      input["premise"]=self.premise[index]
      input["hypothesis"]=self.hypothesis[index]


      """"
      이거는 인풋을 리스트 형태로 저장한것
      input={"premise": [], "hypothesis": []}

      input["premise"].append(self.premise[index])
      input["hypothesis"].append(self.hypothesis[index])
      """

      #print(input)
      tokenized_dataset = tokenizing_function(input)
      tokenized_dataset["labels"]=[]
      tokenized_dataset["labels"].append(self.labels[index])
      #print("tokenized_dataset",tokenized_dataset)

      """""  
      input_ids=[]
      attention_mask=[]
      premise=self.tokenizer(self.premise)
      hypothesis=self.tokenizer(self.hypothesis)
      input_ids.append(premise["input_ids"])
      input_ids.append(hypothesis["input_ids"])
      attention_mask.append(premise["attention_mask"])
      attention_mask.append(hypothesis["attention_mask"])
      labels=self.labels
      """

      return tokenized_dataset


      """""
      {"input_ids" :torch.tensor(input_ids[index], dtype=torch.float64),
      "attention_mask" :torch.tensor(attention_mask[index], dtype=torch.float64),           changing the position tokenizer applying
      "labels":torch.tensor(labels[index],dtype=torch.float64)}
      """

  def __len__(self):
    return self.len

  def collate_fn(self,data):                                    #max기준으로?-O ...batch의 가장 긴 sentence 기준??-X
    #print("data",data)
    #print(len(data))

    input_ids=[dic['input_ids'] for dic in data]
    attention_mask=[dic['attention_mask'] for dic in data]
    labels=[dic['labels'] for dic in data]
    new_input_ids=torch.zeros(len(labels),self.max_seq_len,requires_grad=True).long()
    new_attention_mask=torch.zeros(len(labels),self.max_seq_len,requires_grad=True).long()
    for i,sentence in enumerate(input_ids):
        new_input_ids[i][:min(len(sentence),self.max_seq_len)]=torch.tensor(sentence[:min(len(sentence),self.max_seq_len)])

    for i,sentence in enumerate(attention_mask):
        new_attention_mask[i][:min(len(sentence),self.max_seq_len)]=torch.tensor(sentence[:min(len(sentence),self.max_seq_len)])
    #print("new_input_ids",new_input_ids)
    #print("new_attention_mask",new_attention_mask)
    labels=torch.tensor(labels).view(-1).long()
    #print("labels",labels)
    return {"input_ids": new_input_ids,"attention_mask":new_attention_mask, "labels":labels}        #이렇게 해야 dataloader이 dictionary 반환한다.

