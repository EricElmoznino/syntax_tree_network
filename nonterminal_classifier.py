import argparse
import os
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ignite.engine import Engine
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from training import run
from models import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NonterminalFeaturesDataset(Dataset):

    def __init__(self, data_dir):
        data = os.listdir(data_dir)
        data = [d for d in data if '.pth' in d]
        data = [(os.path.join(data_dir, d), int(d.split('_')[0])) for d in data]
        self.data = data
        self.feature_size = torch.load(data[0][0]).shape[0]
        self.num_nonterminals = max([d[1] for d in data]) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        features, nonterminal = self.data[item]
        features, nonterminal = torch.load(features), torch.LongTensor([nonterminal])
        return features, nonterminal


def step_train(model, optimizer, train=True):
    if train:
        assert optimizer is not None
    else:
        assert optimizer is None

    def step(engine, batch):
        if train:
            model.train()
        else:
            model.eval()

        feats, label = batch
        label = label.unsqueeze(1)
        feats.to(device)
        label.to(device)

        c = classifier(feats)

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
    req_grp.add_argument('--data_dir', required=True, type=str,
                         help='path to the directory of the dataset.')
    req_grp.add_argument('--epochs', default=5, type=int, help='number of epochs.')
    req_grp.add_argument('--batch_size', default=5, type=int, help='batch size.')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    args = parser.parse_args()

    # Create dataset, model, and optimizer
    train_set = NonterminalFeaturesDataset(os.path.join(args.data_dir, 'train'))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_set = NonterminalFeaturesDataset(os.path.join(args.data_dir, 'val'))
    val_loader = DataLoader(train_set, batch_size=args.batch_size)
    gen_set = NonterminalFeaturesDataset(os.path.join(args.data_dir, 'gen'))
    gen_loader = DataLoader(train_set, batch_size=args.batch_size)

    classifier = Classifier(train_set.feature_size, train_set.num_nonterminals)
    classifier = classifier.to(device)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr)

    # Trainer and metrics
    save_dict = {'classifier': classifier}
    trainer = Engine(step_train(classifier, optimizer))
    metric_names = ['loss', 'accuracy']
    RunningAverage(Loss(F.cross_entropy, lambda x: (x['y_pred'], x['y_true']))).attach(trainer, 'loss')
    RunningAverage(Accuracy(lambda x: (x['y_pred'], x['y_true']))).attach(trainer, 'accuracy')

    # Evaluator and metrics
    evaluator = Engine(step_train(classifier, None, train=False))
    Accuracy(lambda x: (x['y_pred'], x['y_true'])).attach(evaluator, 'accuracy')

    # Begin training
    run(args.run_name, save_dict, metric_names, trainer, evaluator,
        train_loader, val_loader, gen_loader, args.epochs, 'accuracy')

    # Display best model final results
    best_model_folder = os.path.join('saved_runs', args.run_name, 'checkpoints')
    files = os.listdir(best_model_folder)
    best_model = [f for f in files if '.pth' in f][0]
    classifier.load_state_dict(torch.load(os.path.join(best_model_folder, best_model), map_location=device))
    ConfusionMatrix(train_set.num_nonterminals, output_transform=lambda x: (x['y_pred'], x['y_true'])).\
        attach(evaluator, 'accuracy')
    print('\nBest model results:')

    evaluator.run(val_loader)
    print('\nValidation:')
    print('Accuracy: {:.2f}'.format(evaluator.state.metrics['accuracy']))
    print('Confusion Matrix:')
    print(evaluator.state.metrics['confusion_matrix'])

    evaluator.run(gen_loader)
    print('\nGeneralization:')
    print('Accuracy: {:.2f}'.format(evaluator.state.metrics['accuracy']))
    print('Confusion Matrix:')
    print(evaluator.state.metrics['confusion_matrix'])
