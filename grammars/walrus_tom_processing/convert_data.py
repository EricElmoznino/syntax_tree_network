import json
from copy import deepcopy


def convert(known_parse_data, tom_data, grammar, save_path):
    v_to_nt = vocabulary_map(grammar)
    tom_sentences, tom_nonterminals = sentence_nonterminals(tom_data, v_to_nt)
    no_vocab_parse_map = parse_map(known_parse_data)
    parses = []
    for s, nt in zip(tom_sentences, tom_nonterminals):
        p = deepcopy(no_vocab_parse_map[nt])
        full_parse(s, p)
        parses.append(p)
    parses = [json.dumps(p) + '\n' for p in parses]
    with open(save_path, 'w') as f:
        f.writelines(parses)


def vocabulary_map(grammar):
    vocab_lines = []
    with open(grammar) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l != '']
    vocab = False
    for l in lines:
        if l == '# Vocabulary':
            vocab = True
            continue
        if not vocab:
            continue
        vocab_lines.append(l)
    v_to_nt = {}
    for v_line in vocab_lines:
        _, nt, v = v_line.split()
        v_to_nt[v] = nt
    return v_to_nt


def sentence_nonterminals(tom_data, v_to_nt):
    with open(tom_data) as f:
        sentences = f.readlines()
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if s != '']
    sentences = [s.split('.')[0].strip().split() for s in sentences]
    nonterminals = [' '.join([v_to_nt[v] for v in s]) for s in sentences]
    return sentences, nonterminals


def recursive_nonterminals(parse):
    if len(parse) == 2:
        return [parse[0]]
    else:
        children = []
        for i in range(1, len(parse)):
            children += recursive_nonterminals(parse[i])
    return children


def recursive_no_vocab(parse):
    for i in range(1, len(parse)):
        if len(parse[i]) == 2:
            parse[i] = [parse[i][0]]
        else:
            recursive_no_vocab(parse[i])


def parse_map(known_parse_data):
    with open(known_parse_data) as f:
        parses = f.readlines()
    parses = [json.loads(p) for p in parses]
    no_vocab_parses = []
    nonterminals = []
    for p in parses:
        nonterminals.append(recursive_nonterminals(p))
        recursive_no_vocab(p)
        no_vocab_parses.append(p)
    return {' '.join(nt): p for nt, p in zip(nonterminals, no_vocab_parses)}


def full_parse(sentence, nonterminal_parse):
    if len(nonterminal_parse) == 1:
        t = sentence.pop(0)
        nonterminal_parse.append(t)
    else:
        for c in nonterminal_parse[1:]:
            full_parse(sentence, c)
