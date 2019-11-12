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


def step_creator(model, decoder, transformer, optimizer, train=True):
    if train:
        assert optimizer is not None
    else:
        assert optimizer is None

    def step(engine, batch):
        if train:
            transformer.train()
        else:
            transformer.eval()

        tree, tree_ques = batch
        if torch.cuda.is_available():
            tree.cuda()
            tree_ques.cuda()

        with torch.no_grad():
            h = model(tree).detach()
        if train:
            pred = decoder(tree_ques, transformer(h))
        else:
            with torch.no_grad():
                pred = decoder(tree_ques, transformer(h))

        true = tree_ques.sentence()
        pred_token = pred.view(-1, pred.shape[-1])
        true_token = true.view(-1)

        pred_first_token = pred[0, :, :]
        true_first_token = true[0, :]

        pred_sent = pred.argmax(dim=-1)
        pred_sent = (pred_sent == true).all(dim=0).long()
        true_sent = torch.ones_like(pred_sent, dtype=torch.long)

        if train:
            loss = F.cross_entropy(pred_token, true_token)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return {'pred_token': pred_token, 'true_token': true_token,
                'pred_first_token': pred_first_token, 'true_first_token': true_first_token,
                'pred_sent': pred_sent, 'true_sent': true_sent}

    return step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    req_grp = parser.add_argument_group('arguments')
    req_grp.add_argument('--run_name', required=True, type=str, help='name of this experiment.')
    req_grp.add_argument('--pretrained_run', required=True, type=str,
                         help='name of the experiment where pretrained encoder/decoder are saved.')
    req_grp.add_argument('--data_dir', default='data/walrus', type=str,
                         help='path to the directory of the dataset.')
    req_grp.add_argument('--model', default='syntax_tree_network',
                         type=str, help='name of the model to compute features.',
                         choices=['syntax_tree_network', 'tree_network', 'syntax_tree_gru', 'tree_gru',
                                  'gru_mixed'])
    req_grp.add_argument('--epochs', default=100, type=int, help='number of epochs.')
    req_grp.add_argument('--batch_size', default=5, type=int, help='batch size.')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    req_grp.add_argument('--hidden_size', default=50, type=int, help='hidden layer size.')
    req_grp.add_argument('--n_transformer_layers', default=3, type=int, help='number of transformer layers.')
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
    model.eval()
    decoder.eval()

    pretrained_dir = os.path.join('saved_runs', args.pretrained_run, 'checkpoints')
    files = os.listdir(pretrained_dir)
    model_file, decoder_file = [f for f in files if 'model' in f][0], [f for f in files if 'decoder' in f][0]
    model.load_state_dict(torch.load(os.path.join(pretrained_dir, model_file), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(pretrained_dir, decoder_file), map_location=device))

    transformer = Transformer(args.hidden_size, args.n_transformer_layers).to(device)

    optimizer = torch.optim.SGD(transformer.parameters(), lr=args.lr)

    # Trainer and metrics
    save_dict = {'transformer': transformer}
    trainer = Engine(step_creator(model, decoder, transformer, optimizer, train=True))
    metric_names = ['loss', 'token_accuracy', 'first_token_accuracy', 'sentence_accuracy']
    RunningAverage(Loss(F.cross_entropy, lambda x: (x['pred_token'], x['true_token']))).attach(trainer, 'loss')
    RunningAverage(Accuracy(lambda x: (x['pred_token'], x['true_token']))).attach(trainer, 'token_accuracy')
    RunningAverage(Accuracy(lambda x: (x['pred_first_token'], x['true_first_token']))) \
        .attach(trainer, 'first_token_accuracy')
    RunningAverage(Accuracy(lambda x: (x['pred_sent'], x['true_sent']))).attach(trainer, 'sentence_accuracy')

    # Evaluator and metrics
    evaluator = Engine(step_creator(model, decoder, transformer, None, train=False))
    Accuracy(lambda x: (x['pred_token'], x['true_token'])).attach(evaluator, 'token_accuracy')
    Accuracy(lambda x: (x['pred_first_token'], x['true_first_token'])).attach(evaluator, 'first_token_accuracy')
    Accuracy(lambda x: (x['pred_sent'], x['true_sent'])).attach(evaluator, 'sentence_accuracy')

    # Begin language modeling training
    run(args.run_name, save_dict, metric_names, trainer, evaluator,
        train_loader, val_loader, gen_loader, args.epochs, 'token_accuracy')
