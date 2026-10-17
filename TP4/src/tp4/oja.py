"""Regla de Oja online para la primera dirección principal."""
from dataclasses import asdict

import numpy as np

from .checkpoint import TrainingCheckpoint, sample_digest
from .config import OjaConfig
from .numerics import matrix, schedule
from .pca import compute_pca
from .results import OjaEpoch, OjaMetrics


class Oja:
    def __init__(self, config=None):
        self.config = config or OjaConfig()

    def fit(self, x, *, checkpoint: TrainingCheckpoint | None = None, on_epoch=None):
        samples = matrix(x)
        if len(samples) < 2:
            raise ValueError('Se requieren al menos dos observaciones')
        rng = self._prepare_training(samples, checkpoint)
        centered = samples - self.mean_
        self._reference = compute_pca(centered, center=False)
        self._total_variance = float(np.var(centered, axis=0, ddof=1).sum())
        while self.epochs_ < self.config.epochs and not self._stopped:
            epoch = self.epochs_ + 1
            self.weights_, record = self._train_epoch(centered, self.weights_, rng, epoch)
            self.history_.append(record)
            self._track_stability(record.direction_change)
            self._stopped = self._should_stop(self._stable_epochs, epoch)
            self.epochs_ = epoch
            self.updates_ = epoch * len(samples)
            if on_epoch is not None:
                on_epoch(self._checkpoint(rng))
        self.stop_reason_ = 'early_stopping' if self._stopped else 'max_epochs'
        return self

    def _prepare_training(self, samples, checkpoint):
        rng = np.random.default_rng(self.config.seed)
        self._data_digest = sample_digest(samples)
        self.history_: list[OjaEpoch] = []
        self.epochs_ = self.updates_ = self._stable_epochs = 0
        self._stopped = False
        self.mean_ = samples.mean(axis=0)
        if checkpoint is None:
            self.weights_ = self._initialize_weights(samples.shape[1], rng)
            return rng
        checkpoint.validate('oja', asdict(self.config), samples)
        self.weights_ = checkpoint.weights.copy()
        self.mean_ = checkpoint.mean.copy()
        self.history_ = list(checkpoint.history)
        self.epochs_ = checkpoint.epoch
        self.updates_ = self.epochs_ * len(samples)
        self._stable_epochs = checkpoint.stable_epochs
        self._stopped = checkpoint.stopped
        rng.bit_generator.state = checkpoint.rng_state
        return rng

    def _track_stability(self, direction_change):
        if direction_change <= self.config.early_stopping.min_delta:
            self._stable_epochs += 1
        else:
            self._stable_epochs = 0

    def _checkpoint(self, rng):
        return TrainingCheckpoint(
            algorithm='oja', parameters=asdict(self.config), sample_digest=self._data_digest,
            epoch=self.epochs_, weights=self.weights_.copy(), rng_state=rng.bit_generator.state,
            history=list(self.history_), stopped=self._stopped, mean=self.mean_.copy(),
            stable_epochs=self._stable_epochs,
        )

    @staticmethod
    def _initialize_weights(feature_count, rng):
        weights = rng.normal(size=feature_count)
        weights /= np.linalg.norm(weights)
        return weights

    def _train_epoch(self, samples, weights, rng, epoch) -> tuple[np.ndarray, OjaEpoch]:
        previous_weights = weights.copy()
        order = rng.permutation(len(samples)) if self.config.shuffle else np.arange(len(samples))
        for step, index in enumerate(order):
            learning_rate = self._learning_rate(epoch, step, len(samples))
            self._update_sample(samples[index], weights, learning_rate)
        norm = self._normalize(weights)
        direction_change = float(np.clip(1 - abs(previous_weights @ weights), 0, 1))
        record = OjaEpoch(
            epoch=epoch,
            updates=epoch * len(samples),
            direction_change=direction_change,
            norm_before_normalization=float(norm),
            learning_rate=learning_rate,
            captured_variance=float(np.var(samples @ weights, ddof=1)),
            pc1_alignment=float(abs(weights @ self._reference.components[0])),
            explained_variance_ratio=float(np.var(samples @ weights, ddof=1) / self._total_variance),
        )
        return weights, record

    def _learning_rate(self, epoch, step, sample_count):
        config = self.config
        update = (epoch - 1) * sample_count + step
        progress = update / max(config.epochs * sample_count - 1, 1)
        return schedule(config.learning_rate_start, config.learning_rate_end, progress, config.decay)

    @staticmethod
    def _update_sample(sample, weights, learning_rate):
        with np.errstate(over='ignore', invalid='ignore'):
            output = sample @ weights
            weights += learning_rate * output * (sample - output * weights)
        if not np.isfinite(weights).all():
            raise FloatingPointError('Oja divergió; disminuir tasa de aprendizaje')

    @staticmethod
    def _normalize(weights):
        norm = np.linalg.norm(weights)
        if not np.isfinite(norm) or norm == 0:
            raise FloatingPointError('Norma inválida en Oja')
        weights /= norm
        return norm

    def _should_stop(self, stable_epochs, epoch):
        config = self.config.early_stopping
        return config.enabled and epoch >= config.min_epochs and stable_epochs >= config.patience

    def _fitted(self):
        if not hasattr(self, 'weights_'):
            raise ValueError('Primero ejecutar fit')

    def transform(self, x):
        self._fitted()
        return (matrix(x, len(self.weights_)) - self.mean_) @ self.weights_

    def metrics(self, x) -> dict:
        """Vista compatible; evaluate devuelve un resultado tipado."""
        return asdict(self.evaluate(x))

    def evaluate(self, x) -> OjaMetrics:
        self._fitted()
        centered = matrix(x, len(self.weights_)) - self.mean_
        reference = compute_pca(centered, center=False)
        score = self.transform(x)
        pc1 = centered @ reference.components[0]
        variance = float(np.var(score, ddof=1))
        total = float(np.var(centered, axis=0, ddof=1).sum())
        if total <= 0 or np.std(score) == 0 or np.std(pc1) == 0:
            raise ValueError('Comparación con PCA requiere variación no nula')
        return OjaMetrics(
            pc1_alignment=float(abs(self.weights_ @ reference.components[0])),
            pc1_score_correlation=float(abs(np.corrcoef(score, pc1)[0, 1])),
            captured_variance=variance,
            explained_variance_ratio=variance / total,
            pc1_variance=float(reference.eigenvalues[0]),
        )
