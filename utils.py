"""
Contains helper preprocessing functions
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer


def load_dataset(path, leave='Title'):
    """"
    loads  dataframe
    Args:
        path: location of dataset
        leave: remove that column
    Returns:
        dataset: dataframe 

    """
    df = pd.read_csv(path)
    if leave is not None:
        df = df.drop(leave, 1)
    df['Class Index'] = df['Class Index'].astype(int) - 1
    classes = df['Class Index'].unique()
    classes_names = {k: v for k, v in zip(sorted(classes), ['World', 'Sports', 'Business', 'Sci/Tech'])}
    df['Class'] = df['Class Index'].replace(classes_names)
    return df


class Vocabulary(Dataset):
    """Build custom dataset for DataLoader
    Helpers:
        .word2index: returns word2index for dataset
        .convert_sequence: converts sequence to corresponding indexes
        .labels_list : returns number of labels in dataset
        .max_seq_len: returns max sequence length of dataset
    """
    def __init__(self, df_train, df_labels):
        self.labels_list = set(df_labels)
        self.word2index, self.tokenizer = self.build_vectorizer(df_train)

        sequences = [self.convert_sequence(sequence, self.word2index, self.tokenizer)
                     for sequence in df_train]
        self.max_seq_len = max([len(seq) for seq in sequences])
        self.sequences = [self.pad_index(sequence, self.max_seq_len, self.word2index)
                          for sequence in sequences]
        self.labels = df_labels

    def build_vectorizer(self, sequences_lists, stop_w='english', min_df=0):
        vectorizer = CountVectorizer(stop_words=stop_w, min_df=min_df)
        vectorizer.fit(sequences_lists)
        word2index = vectorizer.vocabulary_
        word2index['<PAD>'] = max(word2index.values()) + 1
        tokenizer = vectorizer.build_analyzer()
        return word2index, tokenizer

    def convert_sequence(self, sequence, word2index, tokenizer_func):
        """encode a sequence  to a list of indexes"""
        return [word2index[word] for word in tokenizer_func(sequence)
                if word in word2index]

    def pad_index(self, sequence, max_seq_len, word2index, pad_key='<PAD>'):
        """pads a sequence of indexes to max length """
        return sequence + (max_seq_len - len(sequence)) * [word2index[pad_key]]

    def __getitem__(self, i):
        assert len(self.sequences[i]) == self.max_seq_len
        return self.sequences[i], self.labels[i]

    def __len__(self):
        return len(self.sequences)


def build_dataloader(dataset, batch_size, split=False, train_size=None):
    """Build dataloader for torch dataset
    Args:
        dataset: torch dataset
        batch_size: batch size
        split: if True returns train, val data loader
        train_size: if split is True, the size of train_size in decimal (eg 0.8)
    returns:
        dataloader
    """

    def collate(batch):
        inputs = torch.LongTensor([item[0] for item in batch])
        target = torch.LongTensor([item[1] for item in batch])
        return inputs, target

    if split:
        train_size = int(train_size * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)
        return train_loader, val_loader

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)
    return dataloader
