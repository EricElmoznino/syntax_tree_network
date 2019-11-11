import os
import random
import torch
from tensorboardX import SummaryWriter
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar

# Random seeding
random.seed(99)
torch.manual_seed(99)
if torch.cuda.is_available(): torch.cuda.manual_seed(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(run_name, save_dict, metric_names, trainer, evaluator,
        train_loader, val_loader, gen_loader, epochs, save_metric):
    save_dir = os.path.join('saved_runs', run_name)
    os.mkdir(save_dir)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))

    ProgressBar().attach(trainer, metric_names=metric_names)
    checkpoint_handler = ModelCheckpoint(os.path.join(save_dir, 'checkpoints'), '', require_empty=False,
                                         score_function=lambda e: e.state.metrics[save_metric])
    if evaluator is not None:
        evaluator.add_event_handler(event_name=Events.COMPLETED, handler=checkpoint_handler, to_save=save_dict)
    else:
        trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save=save_dict)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_results(engine):
        if engine.state.iteration % 1000:
            for metric, value in engine.state.metrics.items():
                writer.add_scalar("training/{}".format(metric), value, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if val_loader is None:
            return
        evaluator.run(val_loader)
        results = ['Validation Results - Epoch: {}'.format(engine.state.epoch)]
        for metric, value in evaluator.state.metrics.items():
            writer.add_scalar("validation/{}".format(metric), value, engine.state.iteration)
            results.append('{}: {:.2f}'.format(metric, value))
        print(' '.join(results))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_generalization_results(engine):
        if gen_loader is None:
            return
        evaluator.run(gen_loader)
        results = ['Generalization Results - Epoch: {}'.format(engine.state.epoch)]
        for metric, value in evaluator.state.metrics.items():
            writer.add_scalar("generalization/{}".format(metric), value, engine.state.iteration)
            results.append('{}: {:.2f}'.format(metric, value))
        print(' '.join(results))

    @trainer.on(Events.EPOCH_COMPLETED)
    def shuffle_train_set(engine):
        train_loader.dataset.shuffle_data()

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        writer.close()
        engine.terminate()
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            import warnings
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            # checkpoint_handler(engine, {'model_exception': model})
        else:
            raise e

    trainer.run(train_loader, epochs)
    writer.close()
