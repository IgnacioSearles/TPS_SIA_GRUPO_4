"""Geometría de reconstrucciones RGB con referencia; no supervisa al VAE."""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from src.data.art import BACKGROUND, render_art

METRICS_VERSION = 1


def reference_mask(size, record):
    """Silueta a cobertura >= 50 %, independiente del color del objeto."""
    objects = [{**obj, "color_rgb": [1., 1., 1.]} for obj in record["objects"]]
    return render_art(size, objects, (0., 0., 0.))[:, :, 0] >= .5


def silhouette_scores(expected, predicted):
    """IoU, F1 de borde a 1 px y distancia simétrica media de contorno.

    Una salida vacía falla: no se elimina del promedio. Su distancia se fija
    en la diagonal de la imagen. La referencia debe contener una figura.
    """
    expected, predicted = np.asarray(expected, bool), np.asarray(predicted, bool)
    if expected.ndim != 2 or expected.shape != predicted.shape or not expected.any():
        raise ValueError("Se necesitan máscaras 2D alineadas y una referencia no vacía")
    intersection = np.count_nonzero(expected & predicted)
    union = np.count_nonzero(expected | predicted)
    base = {"silhouette_iou": intersection / union, "empty_prediction": not bool(predicted.any())}
    if not predicted.any():
        return {**base, "boundary_f1": 0., "contour_distance_px": float(np.hypot(*expected.shape))}
    structure = np.ones((3, 3), bool)
    edge_expected = expected ^ binary_erosion(expected, structure=structure)
    edge_predicted = predicted ^ binary_erosion(predicted, structure=structure)
    to_expected = distance_transform_edt(~edge_expected)[edge_predicted]
    to_predicted = distance_transform_edt(~edge_predicted)[edge_expected]
    precision, recall = np.mean(to_expected <= 1), np.mean(to_predicted <= 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {**base, "boundary_f1": float(f1),
            "contour_distance_px": float((to_expected.mean() + to_predicted.mean()) / 2)}


def reconstruction_scores(original, reconstructed, record):
    """Segmenta por contraste RGB respecto del fondo conocido.

    El umbral es la mitad del contraste del color de referencia. No se cambia
    por modelo ni se busca el mejor umbral. Color/forma no quedan completamente
    aislados: un objeto muy tenue también puede perder silueta.
    """
    if (original.shape != reconstructed.shape or original.ndim != 3
            or original.shape[2] != 3 or original.shape[0] != original.shape[1]
            or len(record["objects"]) != 1):
        raise ValueError("La evaluación geométrica requiere RGB cuadrado de un objeto")
    if not np.isfinite(reconstructed).all():
        raise ValueError("La reconstrucción contiene valores no finitos")
    expected = reference_mask(original.shape[0], record)
    contrast = np.linalg.norm(np.asarray(record["objects"][0]["color_rgb"]) - BACKGROUND)
    # La media antialias float32 puede desplazar una cobertura exacta de 0,5
    # unas décimas de micro-unidad. No penalizar una copia idéntica por redondeo.
    predicted = np.linalg.norm(reconstructed - BACKGROUND, axis=2) >= .5 * contrast - 1e-6
    core = binary_erosion(expected, iterations=2)
    if not core.any():
        core = expected
    squared = (original - reconstructed)**2
    return {**silhouette_scores(expected, predicted), "mse": float(squared.mean()),
            "foreground_mse": float(squared[expected].mean()),
            "interior_color_mse": float(squared[core].mean())}


def summarize_geometry(rows):
    """Promedio por familia y luego entre familias; conserva todos los casos."""
    metrics = ("silhouette_iou", "boundary_f1", "contour_distance_px", "mse",
               "foreground_mse", "interior_color_mse")
    families = []
    for family in sorted({r["family"] for r in rows}):
        chosen = [r for r in rows if r["family"] == family]
        families.append({"family": family, "samples": len(chosen),
                         **{m: float(np.mean([r[m] for r in chosen])) for m in metrics}})
    result = {"samples": len(rows), "empty_predictions": sum(r["empty_prediction"] for r in rows),
              **{f"macro_{m}": float(np.mean([r[m] for r in families])) for m in metrics}}
    return result, families


def calibrate_geometry(samples, records):
    """Una copia exacta de cada referencia debe tener geometría perfecta."""
    size = int(np.sqrt(samples.shape[1] / 3))
    scores = [reconstruction_scores(a.reshape(size, size, 3), a.reshape(size, size, 3), r)
              for a, r in zip(samples, records)]
    return {"references": len(scores), "minimum_identity_iou": float(min(r["silhouette_iou"] for r in scores)),
            "nonperfect_references": int(sum(r["silhouette_iou"] != 1 or r["boundary_f1"] != 1
                                            or r["contour_distance_px"] != 0 for r in scores))}
