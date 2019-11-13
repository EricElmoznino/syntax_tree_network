import argparse
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ignite.engine import Engine
from ignite.metrics import Accuracy, Loss, RunningAverage
from training import run
from models import *
from datasets import FlyingPlanesDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def step_train(model, optimizer):
    def step(engine, batch):
        model.train()

        tree, label = batch
        if torch.cuda.is_available():
            tree.cuda()
        label.to(device)

        h = model(tree)
        c = classifier(h)

        loss = F.cross_entropy(c, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {'y_pred': c, 'y_true': label}
    return step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    req_grp = parser.add_argument_group('arguments')
    req_grp.add_argument('--run_name', required=True, type=str, help='name of this experiment.')
    req_grp.add_argument('--data_dir', default='data/flying_planes', type=str,
                         help='path to the directory of the dataset.')
    req_grp.add_argument('--model', default='syntax_tree_network',
                         type=str, help='name of the model to compute features.',
                         choices=['syntax_tree_network', 'tree_network', 'rnn'])
    req_grp.add_argument('--epochs', default=5, type=int, help='number of epochs.')
    req_grp.add_argument('--batch_size', default=5, type=int, help='batch size.')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    req_grp.add_argument('--hidden_size', default=20, type=int, help='hidden layer size.')
    args = parser.parse_args()

    # Create dataset, model, and optimizer
    train_set = FlyingPlanesDataset(args.data_dir, args.batch_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=None)

    if args.model == 'syntax_tree_network':
        model = SyntaxTreeNetwork(train_set.input_size, args.hidden_size,
                                  train_set.num_nonterminal_rules, train_set.num_nonterminals)
    elif args.model == 'tree_network':
        model = TreeNetwork(train_set.input_size, args.hidden_size)
    elif args.model == 'rnn':
        model = RNN(train_set.input_size, args.hidden_size)
    else:
        raise NotImplementedError('unknown model type {}'.format(args.model))
    model = model.to(device)

    classifier = Classifier(args.hidden_size, train_set.n_classes)
    classifier = classifier.to(device)

    optimizer = torch.optim.SGD(list(model.parameters()) + list(classifier.parameters()), lr=args.lr)

    # Trainer and metrics
    save_dict = {'model': model, 'classifier': classifier}
    trainer = Engine(step_train(model, optimizer))
    metric_names = ['loss', 'accuracy']
    RunningAverage(Loss(F.cross_entropy, lambda x: (x['y_pred'], x['y_true']))).attach(trainer, 'loss')
    RunningAverage(Accuracy(lambda x: (x['y_pred'], x['y_true']))).attach(trainer, 'accuracy')

    # Begin training
    run(args.run_name, save_dict, metric_names, trainer, None, train_loader, None, None, args.epochs, 'accuracy')
