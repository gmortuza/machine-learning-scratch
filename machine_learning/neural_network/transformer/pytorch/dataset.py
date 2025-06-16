from torch.utils.data import Dataset, DataLoader
from torch import tensor
from transformers import AutoTokenizer
import os
# Import necessary libraries
from utils import get_config, Config

config = get_config()

class TranslationData(Dataset):
    def __init__(self, type: str='train', config: Config = config):
        file_path = config.train_data if type == 'train' else config.valid_data
        self.data = self.read_data(file_path)
        self.src_tokenizer = AutoTokenizer.from_pretrained(config.src_tokenizer)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(config.tgt_tokenizer)
        self.max_length = config.max_seq_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        src_text, tgt_text = self.data[idx]

        # Tokenize source 
        src = self.src_tokenizer(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target 
        tgt = self.tgt_tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels": tgt["input_ids"].squeeze(),
        }

    def read_data(self, file_path: str):
        # print current file path
        config.logger.info(f"Reading data from: {file_path}")
        with open(file_path, "r") as f:
            lines = f.readlines()
        formatted_text = [data.split('\t')[:-1] for data in lines]
        return formatted_text

def get_train_dataloader(config: Config = config):
    dataset = TranslationData(type='train', config=config)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

def get_valid_dataloader(config: Config = config):
    dataset = TranslationData(type='valid', config=config)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

if __name__ == '__main__':
    # test train dataloader
    train_dataloader = get_train_dataloader()
    print(">>>> Testing train dataloader <<<<")
    print(f"Number of batches in train dataloader: {len(train_dataloader.dataset)}")
    for batch in train_dataloader:
        print(batch)
        break  # Just print the first batch
    # test valid dataloader
    print(">>>> Testing valid dataloader <<<<")
    valid_dataloader = get_valid_dataloader()
    print(f"Number of batches in valid dataloader: {len(valid_dataloader.dataset)}")
    for batch in valid_dataloader:
        print(batch)
        break  # Just print the first batch