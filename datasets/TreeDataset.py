import os
import json
import random
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset


class TreeDataset(Dataset):

    def __init__(self, data_dir, batch_size=1, shuffle=False, word_embeddings=None):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.trees = self.read_data(os.path.join(data_dir, 'data.txt'))
        self.unique_trees = self.group_unique_trees()
        self.epoch_batches = self.batches_from_unique()
        self.labels = self.read_labels(os.path.join(data_dir, 'labels.txt'))

        self.word_embeddings = word_embeddings
        self.rule_map, self.non_terminal_map, self.terminal_map = self.scan_dataset()

        if self.word_embeddings is not None:
            self.input_size = len(word_embeddings[list(word_embeddings.keys())[0]])
        else:
            self.input_size = len(self.terminal_map)

    def __len__(self):
        return len(self.epoch_batches)

    def __getitem__(self, item):
        batch_indices = self.epoch_batches[item]
        trees = [self.trees[i] for i in batch_indices]
        tree_tensor = TreeTensor(trees, self.rule_map, self.non_terminal_map, self.terminal_map, self.word_embeddings)
        label = [self.label_tensor(self.labels[i]) for i in batch_indices]
        label = torch.stack(label)
        return tree_tensor, label

    def read_data(self, data_path):
        with open(data_path) as f:
            parses = f.readlines()
        parses = '[' + ','.join(parses) + ']'
        parses = json.loads(parses)
        trees = [Tree(p) for p in parses]
        return trees

    def group_unique_trees(self):
        unique_trees = []
        for i in range(len(self.trees)):
            exists = False
            for unique in unique_trees:
                ref = unique[0]
                if Tree.same_syntax(self.trees[i], self.trees[ref]):
                    unique.append(i)
                    exists = True
                    break
            if not exists:
                unique_trees.append([i])
        return unique_trees

    def batches_from_unique(self):
        batches = []
        for unique in self.unique_trees:
            for i in range(0, len(unique), self.batch_size):
                batches.append(unique[i:i+self.batch_size])
        return batches

    def shuffle_data(self):
        for unique in self.unique_trees:
            random.shuffle(unique)
        self.epoch_batches = self.batches_from_unique()
        random.shuffle(self.epoch_batches)

    def scan_dataset(self):
        unique_rules = set()
        unique_non_terminals = set()
        unique_terminals = set() if self.word_embeddings is None else None
        for tree in self.trees:
            for r in tree.all_rules():
                unique_rules.add(r)
            for nt in tree.all_non_terminals():
                unique_non_terminals.add(nt)
            if self.word_embeddings is None:
                for t in tree.all_terminals():
                    unique_terminals.add(t)
        rule_map = {r: i for i, r in enumerate(list(unique_rules))}
        non_terminal_map = {nt: i for i, nt in enumerate(list(unique_non_terminals))}
        if self.word_embeddings is None:
            terminal_map = {t: i for i, t in enumerate(list(unique_terminals))}
        else:
            terminal_map = None
        return rule_map, non_terminal_map, terminal_map

    def read_labels(self, label_path):
        raise NotImplementedError()

    def label_tensor(self, label):
        raise NotImplementedError()


class TreeTensor:

    def __init__(self, trees, rule_map, non_terminal_map, terminal_map, word_embeddings=None):
        assert isinstance(trees, list)
        ref = trees[0]
        if ref.is_terminal:
            self.is_terminal = True
            self.children = None
            self.rule = None
            if word_embeddings is None:
                terminals = torch.LongTensor([terminal_map[tree.node] for tree in trees])
                terminals = one_hot(terminals, num_classes=len(terminal_map))
            else:
                terminals = torch.stack([torch.FloatTensor(word_embeddings[tree.node]) for tree in trees])
            self.node = terminals
        else:
            self.is_terminal = False
            self.rule = rule_map[ref.rule]
            self.node = non_terminal_map[ref.node]
            self.children = []
            for i in range(len(trees[0].children)):
                c_trees = [tree.children[i] for tree in trees]
                c_tree_tensor = TreeTensor(c_trees, rule_map, non_terminal_map, terminal_map, word_embeddings)
                self.children.append(c_tree_tensor)

    def cuda(self):
        assert torch.cuda.is_available()
        self.node = self.node.cuda()
        if not self.is_terminal:
            self.rule = self.rule.cuda()
            for c in self.children:
                c.cuda()


class Tree:

    def __init__(self, sentence_parse):
        if isinstance(sentence_parse, str):
            self.is_terminal = True
            self.node = sentence_parse
            self.children = None
            self.rule = None
        else:
            self.is_terminal = False
            self.node, rhs = sentence_parse[0], sentence_parse[1:]
            self.children = []
            for parse in rhs:
                self.children.append(Tree(parse))
            self.rule = '%s -> %s' % (self.node, ' '.join(['term' if c.is_terminal else c.node for c in self.children]))

    def all_rules(self):
        if self.is_terminal:
            return []
        rules = [self.rule]
        for c in self.children:
            rules += c.all_rules()
        return rules

    def all_non_terminals(self):
        if self.is_terminal:
            return []
        non_terminals = [self.node]
        for c in self.children:
            non_terminals += c.all_non_terminals()
        return non_terminals

    def all_terminals(self):
        if self.is_terminal:
            return [self.node]
        terminals = []
        for c in self.children:
            terminals += c.all_terminals()
        return terminals

    @staticmethod
    def same_syntax(first, second):
        if first.is_terminal and second.is_terminal:
            return True
        if first.node != second.node:
            return False
        if len(first.children) != len(second.children):
            return False
        for c_first, c_second in zip(first.children, second.children):
            if not Tree.same_syntax(c_first, c_second):
                return False
        return True
