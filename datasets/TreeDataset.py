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
        self.shuffle_data()

        self.word_embeddings = word_embeddings
        self.nonterminal_rule_map, self.terminal_rule_map, self.nonterminal_map, self.terminal_map = self.scan_dataset()

        if self.word_embeddings is not None:
            self.input_size = len(word_embeddings[list(word_embeddings.keys())[0]])
        else:
            self.input_size = len(self.terminal_map)
        self.num_nonterminal_rules = len(self.nonterminal_rule_map)
        self.num_terminal_rules = len(self.terminal_rule_map)
        self.num_nonterminals = len(self.nonterminal_map)

    def __len__(self):
        return len(self.epoch_batches)

    def __getitem__(self, item):
        batch_indices = self.epoch_batches[item]
        trees = [self.trees[i] for i in batch_indices]
        tree_tensor = TreeTensor(trees, self.nonterminal_rule_map, self.terminal_rule_map,
                                 self.nonterminal_map, self.terminal_map, self.word_embeddings)
        return tree_tensor

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
        unique_nonterminal_rules = set()
        unique_terminal_rules = set()
        unique_nonterminals = set()
        unique_terminals = set() if self.word_embeddings is None else None
        for tree in self.trees:
            for r in tree.all_nonterminal_rules():
                unique_nonterminal_rules.add(r)
            for r in tree.all_terminal_rules():
                unique_terminal_rules.add(r)
            for nt in tree.all_nonterminals():
                unique_nonterminals.add(nt)
            if self.word_embeddings is None:
                for t in tree.all_terminals():
                    unique_terminals.add(t)
        nonterminal_rule_map = {r: i for i, r in enumerate(list(unique_nonterminal_rules))}
        terminal_rule_map = {r: i for i, r in enumerate(list(unique_terminal_rules))}
        nonterminal_map = {nt: i for i, nt in enumerate(list(unique_nonterminals))}
        if self.word_embeddings is None:
            terminal_map = {t: i for i, t in enumerate(list(unique_terminals))}
        else:
            terminal_map = None
        return nonterminal_rule_map, terminal_rule_map, nonterminal_map, terminal_map


class TreeTensor:

    def __init__(self, trees, nonterminal_rule_map, terminal_rule_map,
                 nonterminal_map, terminal_map, word_embeddings=None):
        assert isinstance(trees, list)
        ref = trees[0]
        if ref.is_terminal:
            self.is_terminal = True
            self.is_preterminal = False
            self.children = None
            self.rule = None
            if word_embeddings is None:
                terminals = torch.LongTensor([terminal_map[tree.node] for tree in trees])
                terminals = one_hot(terminals, num_classes=len(terminal_map)).float()
            else:
                terminals = torch.stack([torch.FloatTensor(word_embeddings[tree.node]) for tree in trees])
            self.node = terminals
        else:
            self.is_terminal = False
            self.node = nonterminal_map[ref.node]
            self.children = []
            for i in range(len(trees[0].children)):
                c_trees = [tree.children[i] for tree in trees]
                c_tree_tensor = TreeTensor(c_trees, nonterminal_rule_map, terminal_rule_map,
                                           nonterminal_map, terminal_map, word_embeddings)
                self.children.append(c_tree_tensor)
            if self.children[0].is_terminal:
                self.is_preterminal = True
                self.rule = terminal_rule_map[ref.rule]
            else:
                self.is_preterminal = False
                self.rule = nonterminal_rule_map[ref.rule]

    def cuda(self):
        assert torch.cuda.is_available()
        self.node = self.node.cuda()
        if not self.is_terminal:
            self.rule = self.rule.cuda()
            for c in self.children:
                c.cuda()


class Tree:

    def __init__(self, sentence_parse):
        if sentence_parse is None:
            self.is_terminal = False
            self.node = None
            self.children = None
            self.rule = None
            return
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

    def all_nonterminal_rules(self):
        if self.is_terminal or self.children[0].is_terminal:
            return []
        rules = [self.rule]
        for c in self.children:
            rules += c.all_nonterminal_rules()
        return rules

    def all_terminal_rules(self):
        if self.is_terminal:
            return []
        rules = [self.rule] if self.children[0].is_terminal else []
        for c in self.children:
            rules += c.all_terminal_rules()
        return rules

    def all_nonterminals(self):
        if self.is_terminal:
            return []
        nonterminals = [self.node]
        for c in self.children:
            nonterminals += c.all_nonterminals()
        return nonterminals

    def all_terminals(self):
        if self.is_terminal:
            return [self.node]
        terminals = []
        for c in self.children:
            terminals += c.all_terminals()
        return terminals

    def sentence(self):
        def recursive_words(t):
            if t.is_terminal:
                return [t.node]
            w = []
            for c in t.children:
                w += recursive_words(c)
            return w
        words = ' '.join(recursive_words(self))
        return words

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
