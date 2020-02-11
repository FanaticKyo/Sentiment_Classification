"""
Sentiment classifier based on RNN

Author: Yohan Jo
Date: October 28, 2019
"""
import torch
from torch import optim, nn
from nltk.tokenize import word_tokenize
import numpy as np
from random import randint
from collections import defaultdict, Counter
import argparse


OUT_HELDOUT_PATH = "heldout_pred_nn.txt"

idx2label = ["positive", "neutral", "negative"]
label2idx = {label: idx for idx, label in enumerate(idx2label)}

class ClassifierRunner(object):
    def __init__(self, data, voca_size, rnn_in_dim, rnn_hid_dim):
        self.data = data
        self.clf = Classifier(voca_size, rnn_in_dim, rnn_hid_dim)
        self.optimizer = optim.Adam(self.clf.parameters())
        self.ce_loss = nn.CrossEntropyLoss()

    def run_epoch(self, split):
        """Runs an epoch, during which the classifier is trained or applied
        on the data. Returns the predicted labels of the instances."""

        if split == "dev": self.clf.train()
        else: self.clf.eval()

        labels_pred = []
        for i, (words, label) in enumerate(self.data[split]):
            logit = self.clf(torch.LongTensor(words))

            # Optimize
            if split == "dev":
                loss = self.ce_loss(logit, torch.LongTensor([label]))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            labels_pred.append(idx2label[randint(0, 2)])

        return labels_pred


class Classifier(nn.Module):
    def __init__(self, voca_size, rnn_in_dim, rnn_hid_dim):
        super(Classifier, self).__init__()

        self.rnn_in_dim = rnn_in_dim
        self.rnn_hid_dim = rnn_hid_dim

        # Layers
        self.word2wemb = nn.Embedding(voca_size, rnn_in_dim)
        self.rnn = nn.RNN(rnn_in_dim, rnn_hid_dim)
        self.rnn2logit = nn.Linear(rnn_hid_dim, 3)

    def init_rnn_hid(self):
        """Initial hidden state."""
        return torch.zeros(1, 1, self.rnn_hid_dim)

    def forward(self, words):
        """Feeds the words into the neural network and returns the value
        of the output layer."""
        wembs = self.word2wemb(words) # (seq_len, rnn_in_dim)
        rnn_outs, _ = self.rnn(wembs.unsqueeze(1), self.init_rnn_hid()) 
                                      # (seq_len, 1, rnn_hid_dim)
        logit = self.rnn2logit(rnn_outs[-1]) # (1 x 3)
        return logit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rnn_in_dim", default=100, type=float, 
                        help="Dimentionality of RNN inputs")
    parser.add_argument("-rnn_hid_dim", default=30, type=float, 
                        help="Dimentionality of RNN hidden state")
    parser.add_argument("-epochs", default=10, type=int,
                        help="Number of epochs")
    args = parser.parse_args()

    print("Reading data...")
    data_raw = defaultdict(list)
    voca_cnt = Counter()
    for text, label in zip(open("dev_text.txt"), open("dev_label.txt")):
        words = word_tokenize(text.strip())
        data_raw["dev"].append((words, label2idx[label.strip()]))
        voca_cnt.update(words)

    for text in open("heldout_text.txt"):
        words = word_tokenize(text.strip())
        data_raw["heldout"].append((words, None))


    print("Building voca...")
    word_idx = {"[UNK]": 0}
    for word in voca_cnt.keys():
        word_idx[word] = len(word_idx)
    print("n_voca:", len(word_idx))

    print("Indexing words...")
    data = defaultdict(list)
    for split in ["dev", "heldout"]:
        for words, label in data_raw[split]:
            data[split].append(([word_idx.get(w, 0) for w in words], label))

    print("Running classifier...")
    M = ClassifierRunner(data, len(word_idx), args.rnn_in_dim, args.rnn_hid_dim)
    for epoch in range(args.epochs):
        print("Epoch", epoch+1)

        # Train
        M.run_epoch("dev")

        # Test
        with torch.no_grad():
            labels_pred = M.run_epoch("heldout")
        with open(OUT_HELDOUT_PATH, "w") as f:
            f.write("\n".join(labels_pred))
