"""SOM online: bucles secuenciales por muestra, neuronas vectorizadas."""
from dataclasses import asdict

import numpy as np

from .checkpoint import TrainingCheckpoint, sample_digest
from .config import KohonenConfig
from .geometry import Grid
from .numerics import Patience, distances, matrix, schedule
from .topology import Topology, create_topology
from .results import KohonenEpoch, KohonenMetrics


class Kohonen:
    def __init__(self, config=None, topology: Topology | None = None):
        self.config = config or KohonenConfig()
        layout = topology if topology is not None else create_topology(self.config.topology)
        self.grid = Grid(self.config.rows, self.config.cols, layout)

    def _fitted(self):
        if not hasattr(self, 'weights_'):
            raise ValueError('Primero ejecutar fit')

    def predict(self, x):
        self._fitted()
        samples = matrix(x, self.weights_.shape[1])
        return distances(samples, self.weights_).argmin(axis=1)

    def metrics(self, x) -> dict:
        """Vista compatible para consumidores que necesitan un diccionario."""
        return asdict(self.evaluate(x))

    def evaluate(self, x) -> KohonenMetrics:
        self._fitted()
        samples = matrix(x, self.weights_.shape[1])
        sample_distances = distances(samples, self.weights_)
        winners = np.argsort(sample_distances, axis=1, kind='stable')[:, :2]
        adjacent = self.grid.adjacency[winners[:, 0], winners[:, 1]]
        return KohonenMetrics(
            quantization_error=float(sample_distances.min(axis=1).mean()),
            topographic_error=float((~adjacent).mean()),
        )

    def fit(self, x, *, checkpoint: TrainingCheckpoint | None = None, on_epoch=None):
        samples = matrix(x)
        rng, stopper = self._prepare_training(samples, checkpoint)
        while self.epochs_ < self.config.epochs and not self._stopped:
            epoch = self.epochs_ + 1
            record = self._train_epoch(samples, rng, epoch)
            self.history_.append(record)
            self._track_best(record.quantization_error, epoch)
            self._stopped = stopper.update(record.quantization_error, epoch)
            self.epochs_ = epoch
            self.updates_ = epoch * len(samples)
            if on_epoch is not None:
                on_epoch(self._checkpoint(rng, stopper))
        self.stop_reason_ = 'early_stopping' if self._stopped else 'max_epochs'
        self.selected_epoch_ = self.epochs_
        if self.config.early_stopping.enabled:
            self.weights_ = self._best_weights.copy()
            self.selected_epoch_ = self.best_epoch_
        return self

    def _prepare_training(self, samples, checkpoint):
        rng = np.random.default_rng(self.config.seed)
        stopper = Patience(self.config.early_stopping)
        self._data_digest = sample_digest(samples)
        self.epochs_ = self.updates_ = 0
        self._stopped = False
        self.history_: list[KohonenEpoch] = []
        self._best_error = float('inf')
        if checkpoint is None:
            self.weights_ = self._initialize_weights(samples, rng)
            return rng, stopper
        checkpoint.validate('kohonen', self._parameters(), samples)
        self.weights_ = checkpoint.weights.copy()
        self.history_ = list(checkpoint.history)
        self.epochs_ = checkpoint.epoch
        self.updates_ = self.epochs_ * len(samples)
        self._stopped = checkpoint.stopped
        self._best_error = checkpoint.best_error
        self._best_weights = checkpoint.best_weights.copy()
        self.best_epoch_ = checkpoint.best_epoch
        rng.bit_generator.state = checkpoint.rng_state
        stopper.reference = checkpoint.patience_reference
        stopper.bad_epochs = checkpoint.bad_epochs
        return rng, stopper

    def _parameters(self):
        return {**asdict(self.config), 'layout': self.grid.topology}

    def _track_best(self, error, epoch):
        if error < self._best_error:
            self._best_error = error
            self._best_weights = self.weights_.copy()
            self.best_epoch_ = epoch

    def _checkpoint(self, rng, stopper):
        return TrainingCheckpoint(
            algorithm='kohonen', parameters=self._parameters(), sample_digest=self._data_digest,
            epoch=self.epochs_, weights=self.weights_.copy(), rng_state=rng.bit_generator.state,
            history=list(self.history_), stopped=self._stopped,
            best_weights=self._best_weights.copy(), best_error=self._best_error,
            best_epoch=self.best_epoch_, patience_reference=stopper.reference,
            bad_epochs=stopper.bad_epochs,
        )

    def _initialize_weights(self, samples, rng):
        count = self.config.rows * self.config.cols
        if self.config.initialization == 'samples':
            indices = rng.choice(len(samples), count, replace=count > len(samples))
            return samples[indices].copy()
        shape = (count, samples.shape[1])
        return rng.uniform(samples.min(axis=0), samples.max(axis=0), shape)

    def _train_epoch(self, samples, rng, epoch) -> KohonenEpoch:
        previous_weights = self.weights_.copy()
        order = rng.permutation(len(samples)) if self.config.shuffle else np.arange(len(samples))
        for step, index in enumerate(order):
            learning_rate, radius = self._step_parameters(epoch, step, len(samples))
            self._update_sample(samples[index], learning_rate, radius)
        evaluation = self.evaluate(samples)
        return KohonenEpoch(
            epoch=epoch,
            updates=epoch * len(samples),
            quantization_error=evaluation.quantization_error,
            topographic_error=evaluation.topographic_error,
            weight_change=float(np.linalg.norm(self.weights_ - previous_weights)),
            learning_rate=learning_rate,
            radius=radius,
        )

    def _step_parameters(self, epoch, step, sample_count):
        config = self.config
        update = (epoch - 1) * sample_count + step
        schedule_epochs = config.schedule_epochs or config.epochs
        progress = min(update / max(schedule_epochs * sample_count - 1, 1), 1.0)
        learning_rate = schedule(
            config.learning_rate_start, config.learning_rate_end, progress, config.decay,
        )
        radius = schedule(config.radius_start, config.radius_end, progress, config.radius_decay)
        return learning_rate, radius

    def neighborhood(self, winner, radius):
        """Peso de la actualización de cada neurona cuando gana `winner` con este radio."""
        grid_distances = self.grid.distances[winner]
        if self.config.neighborhood == 'gaussian':
            return np.exp(-grid_distances * grid_distances / (2 * radius * radius))
        return (grid_distances <= radius).astype(float)

    def _update_sample(self, sample, learning_rate, radius):
        winner = np.square(self.weights_ - sample).sum(axis=1).argmin()
        influence = self.neighborhood(winner, radius)
        self.weights_ += learning_rate * influence[:, None] * (sample - self.weights_)
        if not np.isfinite(self.weights_).all():
            raise FloatingPointError('Kohonen divergió; revisar aprendizaje y datos')

    def occupancy(self, x):
        winners = self.predict(x)
        counts = np.bincount(winners, minlength=len(self.weights_))
        return counts.reshape(self.grid.rows, self.grid.cols)

    def u_matrix(self):
        self._fitted()
        return self.grid.u_matrix(self.weights_)
