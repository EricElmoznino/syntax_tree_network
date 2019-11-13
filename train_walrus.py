import argparse
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ignite.engine import Engine
from ignite.metrics import Accuracy, Loss, RunningAverage
from training import run
from models import *
from datasets import WalrusQuestionDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def step_creator(model, decoder, optimizer, train=True):
    if train:
        assert optimizer is not None
    else:
        assert optimizer is None

    def step(engine, batch):
        if train:
            model.train()
            decoder.train()
        else:
            model.eval()
            decoder.eval()

        tree, tree_ques = batch
        if torch.cuda.is_available():
            tree.cuda()
            tree_ques.cuda()

        if train:
            h_decl, h_ques = model(tree), model(tree_ques)
            pred_decl, pred_ques = decoder(tree, h_decl), decoder(tree_ques, h_ques)
        else:
            with torch.no_grad():
                h_decl, h_ques = model(tree), model(tree_ques)
                pred_decl, pred_ques = decoder(tree, h_decl), decoder(tree_ques, h_ques)

        true_decl, true_ques = tree.sentence(), tree_ques.sentence()
        pred_token_decl = pred_decl.view(-1, pred_decl.shape[-1])
        pred_token_ques = pred_ques.view(-1, pred_ques.shape[-1])
        true_token_decl = true_decl.view(-1)
        true_token_ques = true_ques.view(-1)

        pred_sent_decl, pred_sent_ques = pred_decl.argmax(dim=-1), pred_ques.argmax(dim=-1)
        pred_sent_decl = (pred_sent_decl == true_decl).all(dim=0).long()
        pred_sent_ques = (pred_sent_ques == true_ques).all(dim=0).long()
        true_sent_decl = torch.ones_like(pred_sent_decl, dtype=torch.long)
        true_sent_ques = torch.ones_like(pred_sent_ques, dtype=torch.long)

        if train:
            loss_decl = F.cross_entropy(pred_token_decl, true_token_decl)
            loss_ques = F.cross_entropy(pred_token_ques, true_token_ques)
            loss = loss_decl + loss_ques

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return {'pred_token_decl': pred_token_decl, 'pred_token_ques': pred_token_ques,
                'true_token_decl': true_token_decl, 'true_token_ques': true_token_ques,
                'pred_sent_decl': pred_sent_decl, 'pred_sent_ques': pred_sent_ques,
                'true_sent_decl': true_sent_decl, 'true_sent_ques': true_sent_ques}

    return step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    req_grp = parser.add_argument_group('arguments')
    req_grp.add_argument('--run_name', required=True, type=str, help='name of this experiment.')
    req_grp.add_argument('--data_dir', default='data/walrus', type=str,
                         help='path to the directory of the dataset.')
    req_grp.add_argument('--model', default='syntax_tree_network',
                         type=str, help='name of the model to compute features.',
                         choices=['syntax_tree_network', 'tree_network', 'syntax_tree_gru', 'tree_gru',
                                  'gru_mixed'])
    req_grp.add_argument('--epochs', default=100, type=int, help='number of epochs.')
    req_grp.add_argument('--batch_size', default=5, type=int, help='batch size.')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    req_grp.add_argument('--hidden_size', default=5, type=int, help='hidden layer size.')
    args = parser.parse_args()

    # Create datasets, models, and optimizer
    train_set = WalrusQuestionDataset(os.path.join(args.data_dir, 'train'), args.batch_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=None)
    val_set = WalrusQuestionDataset(os.path.join(args.data_dir, 'val'), args.batch_size)
    val_loader = DataLoader(val_set, batch_size=None)
    gen_set = WalrusQuestionDataset(os.path.join(args.data_dir, 'gen'), args.batch_size)
    gen_loader = DataLoader(gen_set, batch_size=None)

    if args.model == 'syntax_tree_network':
        model = SyntaxTreeNetwork(train_set.input_size, args.hidden_size,
                                  train_set.num_nonterminal_rules, train_set.num_nonterminals)
        decoder = DecoderSyntaxTreeNetwork(train_set.input_size, args.hidden_size, train_set.num_nonterminal_rules,
                                           train_set.num_terminal_rules, train_set.num_nonterminals)
    elif args.model == 'tree_network':
        model = TreeNetwork(train_set.input_size, args.hidden_size)
        decoder = DecoderTreeNetwork(train_set.input_size, args.hidden_size)
    elif args.model == 'syntax_tree_gru':
        model = SyntaxTreeGRU(train_set.input_size, args.hidden_size,
                              train_set.num_nonterminal_rules, train_set.num_nonterminals)
        decoder = DecoderSyntaxTreeGRU(train_set.input_size, args.hidden_size, train_set.num_nonterminal_rules,
                                       train_set.num_terminal_rules, train_set.num_nonterminals)
    elif args.model == 'tree_gru':
        model = TreeGRU(train_set.input_size, args.hidden_size)
        decoder = DecoderTreeGRU(train_set.input_size, args.hidden_size)
    elif args.model == 'gru_mixed':
        model = SyntaxTreeGRU(train_set.input_size, args.hidden_size,
                              train_set.num_nonterminal_rules, train_set.num_nonterminals)
        decoder = DecoderTreeGRU(train_set.input_size, args.hidden_size)
    else:
        raise NotImplementedError('unknown model type {}'.format(args.model))
    model, decoder = model.to(device), decoder.to(device)

    optimizer = torch.optim.SGD(list(model.parameters()) + list(decoder.parameters()), lr=args.lr)

    # Trainer and metrics
    save_dict = {'model': model, 'decoder': decoder}
    trainer = Engine(step_creator(model, decoder, optimizer, train=True))
    metric_names = ['loss_decl', 'loss_ques', 'token_accuracy_decl', 'token_accuracy_ques',
                    'sentence_accuracy_decl', 'sentence_accuracy_ques']
    RunningAverage(Loss(F.cross_entropy, lambda x: (x['pred_token_decl'], x['true_token_decl'])))\
        .attach(trainer, 'loss_decl')
    RunningAverage(Loss(F.cross_entropy, lambda x: (x['pred_token_ques'], x['true_token_ques'])))\
        .attach(trainer, 'loss_ques')
    RunningAverage(Accuracy(lambda x: (x['pred_token_decl'], x['true_token_decl'])))\
        .attach(trainer, 'token_accuracy_decl')
    RunningAverage(Accuracy(lambda x: (x['pred_token_ques'], x['true_token_ques'])))\
        .attach(trainer, 'token_accuracy_ques')
    RunningAverage(Accuracy(lambda x: (x['pred_sent_decl'], x['true_sent_decl'])))\
        .attach(trainer, 'sentence_accuracy_decl')
    RunningAverage(Accuracy(lambda x: (x['pred_sent_ques'], x['true_sent_ques'])))\
        .attach(trainer, 'sentence_accuracy_ques')

    # Evaluator and metrics
    evaluator = Engine(step_creator(model, decoder, None, train=False))
    Accuracy(lambda x: (x['pred_token_decl'], x['true_token_decl'])).attach(evaluator, 'token_accuracy_decl')
    Accuracy(lambda x: (x['pred_token_ques'], x['true_token_ques'])).attach(evaluator, 'token_accuracy_ques')
    Accuracy(lambda x: (x['pred_sent_decl'], x['true_sent_decl'])).attach(evaluator, 'sentence_accuracy_decl')
    Accuracy(lambda x: (x['pred_sent_ques'], x['true_sent_ques'])).attach(evaluator, 'sentence_accuracy_ques')

    # Begin language modeling training
    run(args.run_name, save_dict, metric_names, trainer, evaluator,
        train_loader, val_loader, gen_loader, args.epochs, 'token_accuracy_decl')
