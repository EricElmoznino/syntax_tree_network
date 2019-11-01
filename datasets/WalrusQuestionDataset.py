import json
from copy import deepcopy
from .TreeDataset import TreeDataset, TreeTensor, Tree


class WalrusQuestionDataset(TreeDataset):

    def __init__(self, data_dir, batch_size=1, shuffle=False, word_embeddings=None):
        super().__init__(data_dir, batch_size, shuffle, word_embeddings)

        self.nonterminal_rule_map['SQ -> Aux S'] = len(self.nonterminal_rule_map)
        self.nonterminal_rule_map['S -> NP V'] = len(self.nonterminal_rule_map)
        self.num_nonterminal_rules += 2
        self.nonterminal_map['SQ'] = len(self.nonterminal_map)
        self.num_nonterminals += 1

        self.trees_ques = [generate_question(t) for t in self.trees]

    def __getitem__(self, item):
        tree_tensor = super().__getitem__(item)
        batch_indices = self.epoch_batches[item]
        ques_trees = [self.trees_ques[i] for i in batch_indices]
        tree_ques_tensor = TreeTensor(ques_trees, self.nonterminal_rule_map, self.terminal_rule_map,
                                      self.nonterminal_map, self.terminal_map, self.word_embeddings)
        return tree_tensor, tree_ques_tensor

    def read_data(self, data_path):
        with open(data_path) as f:
            parses = f.readlines()
        parses = '[' + ','.join(parses) + ']'
        parses = json.loads(parses)
        for p in parses:
            p = map_parse(p)
        trees = [Tree(p) for p in parses]
        return trees


def generate_question(tree):
    decl = deepcopy(tree)
    aux = decl.children[1].children[0]
    decl.children[1] = decl.children[1].children[1]
    decl.rule = '%s -> %s' % (decl.node, ' '.join(['term' if c.is_terminal else c.node for c in decl.children]))

    ques = Tree(None)
    ques.node = 'SQ'
    ques.rule = 'SQ -> Aux S'
    ques.children = [aux, decl]

    return ques


def map_parse(sp):
    if isinstance(sp, str):
        return sp

    for i in range(len(sp)):
        sp[i] = map_parse(sp[i])

    # [Vi, ...] -> [V, ...] and [Vt, ...] -> [V, ...]
    if sp[0] == 'Vi' or sp[0] == 'Vt':
        sp[0] = 'V'

    # [VP, [Aux, ...], [V, ...], [NP, ...]] -> [VP, [Aux, ], [VP, [V, ...], [NP, ...]]]
    if len(sp) == 4 and sp[0] == 'VP' and sp[1][0] == 'Aux' and sp[2][0] == 'V' and sp[3][0] == 'NP':
        sp = ['VP', sp[1], ['VP', sp[2], sp[3]]]

    # [NP, [Det, ...], [N, ...], [RC, ...]] -> [NP, [NP, [Det, ...], [N, ...]], [RC, ...]]
    if len(sp) == 4 and sp[0] == 'NP' and sp[1][0] == 'Det' and sp[2][0] == 'N' and sp[3][0] == 'RC':
        sp = ['NP', ['NP', sp[1], sp[2]], sp[3]]

    # [NP, [Det, ...], [N, ...], [P, ...], [Det, ...], [N, ...]] ->
    # [NP, [NP, [Det, ...], [N, ...]], [PP, [P, ...], [NP, [Det, ...], [N, ...]]]]
    if len(sp) == 6 and sp[0] == 'NP' and sp[1][0] == 'Det' and sp[2][0] == 'N' and \
            sp[3][0] == 'P' and sp[4][0] == 'Det' and sp[5][0] == 'N':
        sp = ['NP', ['NP', sp[1], sp[2]], ['PP', sp[3], ['NP', sp[4], sp[5]]]]

    # [RC, [Rel, ...], [Aux, ...], [V, ...]] -> [RC, [Rel, ], [VP, [Aux, ...], [V, ...]]]
    if len(sp) == 4 and sp[0] == 'RC' and sp[1][0] == 'Rel' and sp[2][0] == 'Aux' and sp[3][0] == 'V':
        sp = ['RC', sp[1], ['VP', sp[2], sp[3]]]

    # [RC, [Rel, ...], [Aux, ...], [V, ...], [Det, ...], [N, ...]] ->
    # [RC, [Rel, ], [VP, [Aux, ...], [VP, [V, ...], [NP, [Det, ...], [N, ...]]]]]
    if len(sp) == 6 and sp[0] == 'RC' and sp[1][0] == 'Rel' and sp[2][0] == 'Aux' and sp[3][0] == 'V' and \
            sp[4][0] == 'Det' and sp[5][0] == 'N':
        sp = ['RC', sp[1], ['VP', sp[2], ['VP', sp[3], ['NP', sp[4], sp[5]]]]]

    # [RC, [Rel, ...], [Det, ...], [N, ...], [Aux, ...], [V, ...]] ->
    # [RC, [Rel, ], [S, [NP, [Det, ...], [N, ...]], [VP, [Aux, ...], [V, ...]]]]
    if len(sp) == 6 and sp[0] == 'RC' and sp[1][0] == 'Rel' and sp[2][0] == 'Det' and sp[3][0] == 'N' and \
            sp[4][0] == 'Aux' and sp[5][0] == 'V':
        sp = ['RC', sp[1], ['S', ['NP', sp[2], sp[3]], ['VP', sp[4], sp[5]]]]

    return sp
