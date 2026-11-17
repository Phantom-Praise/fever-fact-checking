"""CPU-only embedding subprocess; isolates Torch from FAISS's OpenMP runtime."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    texts=json.loads(Path(args.input).read_text());torch.set_num_threads(1)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,use_safetensors=True).to('cpu').eval()
    result=[]
    for start in range(0,len(texts),16):
        batch=texts[start:start+16]
        if any(len(tokenizer.encode(t,add_special_tokens=True))>256 for t in batch):raise ValueError('Embedding input exceeds 256-token cap')
        encoded=tokenizer(batch,padding=True,truncation=False,return_tensors='pt')
        with torch.inference_mode():
            hidden=model(**encoded).last_hidden_state;mask=encoded['attention_mask'].unsqueeze(-1)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
            result.append(torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy().astype('float32'))
    np.save(args.output,np.concatenate(result) if result else np.empty((0,384),dtype='float32'),allow_pickle=False)

if __name__=='__main__':main()
