"""Fixed-budget refit on all digits.csv, followed by one external test evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.digit_dataset_loader import load_digit_arrays
from experiments.config import config_from_cli, deep_merge
from experiments.digits.baseline import DEFAULTS, digit_metrics, one_hot
from nn.network import build_model, load_weights, save_weights
from nn.registry import build
from training.callbacks import Callback, ProgressPrinter
from training.trainer import train

FINAL_DEFAULTS = deep_merge(DEFAULTS, {
    "data": {"path": "datasets/digits.csv", "test_path": "datasets/digits_test.csv"},
    "model": {"layers": [784, 64, 32, 10]},
    "optimizer": {"name": "momentum", "lr": 0.1, "momentum": 0.9},
    "training": {"epochs": 76, "batch_size": 128, "shuffle_seed": 42},
    "output": {"directory": "results/digits_final"},
})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TrainRecorder(Callback):
    """Only training metrics; no validation selection or test access."""
    def __init__(self, net, X, labels):
        self.net, self.X, self.labels = net, X, labels

    def on_epoch_end(self, logs):
        outputs = self.net.forward(self.X)
        if not np.isfinite(outputs).all() or not np.isfinite(logs['loss']):
            raise FloatingPointError("Non-finite training values")
        logs['train_accuracy'] = float(np.mean(outputs.argmax(axis=1) == self.labels))


def write_plots(history, metrics, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ['loss', 'train_accuracy'], ['Error de entrenamiento', 'Accuracy de entrenamiento']):
        ax.plot([h['epoch'] for h in history], [h[key] for h in history])
        ax.set(xlabel='Época', title=title)
        ax.grid(alpha=.2)
    fig.suptitle('Entrenamiento final — presupuesto fijo; test no se mide por época')
    fig.tight_layout();fig.savefig(output/'training_curves.png', dpi=150);plt.close(fig)
    matrix = np.array(metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='Blues')
    for actual in range(10):
        for predicted in range(10):
            ax.text(predicted,actual,str(matrix[actual,predicted]),ha='center',va='center',
                    color='white' if matrix[actual,predicted] > matrix.max()/2 else 'black')
    ax.set(xticks=range(10),yticks=range(10),xlabel='Dígito predicho',ylabel='Dígito real',title='Test — modelo final de 76 épocas' if len(history)==76 else 'Test — modelo de presupuesto fijo')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(output/'test_confusion_matrix.png',dpi=150);plt.close(fig)


def run_final(config):
    training_path = Path(config['data']['path'])
    test_path = Path(config['data']['test_path'])
    if training_path.name != 'digits.csv' or test_path.name != 'digits_test.csv' or training_path.resolve() == test_path.resolve():
        raise ValueError('Expected separate digits.csv for refit and digits_test.csv for final evaluation')
    if config['loss'] != 'mse' or config['model']['layers'][0] != 784 or config['model']['layers'][-1] != 10:
        raise ValueError('Expected MSE, 784 inputs and 10 outputs')
    output = Path(config['output']['directory'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new output directory; previous final results must be preserved')
    output.mkdir(parents=True,exist_ok=True)
    # Persist all choices BEFORE loading test or updating weights.
    (output/'config.json').write_text(json.dumps(config,indent=2,allow_nan=False)+'\n')
    protocol = {'mode':'full-data refit with fixed epochs; then test',
                'epochs_fixed_before_test':config['training']['epochs'],
                'epoch_rule':'median of five selected development epochs for momentum 0.9',
                'seed_rule':'original seed 42 retained by convention, not chosen for best validation',
                'no_test_epoch_selection':True,'configuration_sha256':digest(output/'config.json')}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    X, labels = load_digit_arrays(training_path)
    net = build_model(config['model'],np.random.default_rng(config['seed']))
    history = train(net,build('loss',config['loss']),build('optimizer',config['optimizer']),
                    X,one_hot(labels),config['training']['epochs'],config['training']['batch_size'],
                    np.random.default_rng(config['training']['shuffle_seed']),
                    callbacks=[TrainRecorder(net,X,labels),ProgressPrinter(every=10)])
    train_outputs = net.forward(X)
    train_metrics = digit_metrics(labels,train_outputs)
    weights = save_weights(net,output/'final_weights.npz')
    weights_digest = digest(weights)
    pd.DataFrame(history).to_csv(output/'history.csv',index=False)
    restored = load_weights(build_model(config['model'],np.random.default_rng(0)),weights)
    np.testing.assert_array_equal(restored.forward(X[:128]),train_outputs[:128])
    print('Training complete; fixed model saved. Loading external test for final evaluation.',flush=True)
    X_test, test_labels = load_digit_arrays(test_path)
    outputs = restored.forward(X_test)
    metrics = digit_metrics(test_labels,outputs)
    pd.DataFrame({'row_index':np.arange(len(test_labels)),'label':test_labels,'predicted':outputs.argmax(axis=1),
                  **{f'output_{d}':outputs[:,d] for d in range(10)}}).to_csv(output/'test_predictions.csv',index=False)
    training_counts = np.bincount(labels,minlength=10)
    supported = np.isin(test_labels,np.flatnonzero(training_counts))
    # Descriptive coverage analysis AFTER primary all-test evaluation; never drops rows from primary metrics.
    known_metrics = digit_metrics(test_labels[supported],outputs[supported]) if supported.any() else None
    hashes = {hashlib.sha256(row.tobytes()).digest() for row in X}
    overlap = sum(hashlib.sha256(row.tobytes()).digest() in hashes for row in X_test)
    assert digest(weights) == weights_digest
    summary = {'config_sha256':protocol['configuration_sha256'],'weights_sha256':weights_digest,
               'training_dataset_sha256':digest(training_path),'test_dataset_sha256':digest(test_path),
               'train_samples':len(labels),'test_samples':len(test_labels),
               'train_counts':training_counts.tolist(),'test_counts':np.bincount(test_labels,minlength=10).tolist(),
               'missing_training_classes':np.flatnonzero(training_counts==0).tolist(),
               'training_epochs':len(history),'weights_reload_verified':True,
               'test_used_only_after_training':True,'test_model_evaluations':1,
               'train':train_metrics,'test':metrics,
               'diagnostic_test_classes_seen_in_training':{'samples':int(supported.sum()),'metrics':known_metrics},
               'exact_test_images_also_in_training':int(overlap)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    write_plots(history,metrics,output)
    print(f"Final test accuracy: {metrics['accuracy']:.4%}; macro F1: {metrics['macro_f1']:.6f}",flush=True)
    return summary


if __name__ == '__main__':
    run_final(config_from_cli(FINAL_DEFAULTS))
