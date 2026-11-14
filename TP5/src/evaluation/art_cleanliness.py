"""Diagnósticos de limpieza del generador geométrico; no forman parte del error."""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label

from src.data.art import BACKGROUND


def cleanliness(image):
    delta = image - BACKGROUND
    contrast = np.linalg.norm(delta, axis=2)
    brightest = np.argsort(contrast.ravel())[-max(8, image.shape[0]*image.shape[1]//50):]
    color = np.median(image.reshape(-1,3)[brightest], axis=0)
    direction = color - BACKGROUND
    strength = float(np.linalg.norm(direction))
    alpha = np.clip(np.sum(delta*direction,axis=2)/max(strength**2,1e-12),0,1)
    mask = alpha >= .5
    groups, count = label(mask, structure=np.ones((3,3)))
    areas = np.bincount(groups.ravel())[1:]
    largest = float(areas.max()/max(mask.sum(),1)) if len(areas) else 0.
    edge = mask ^ binary_erosion(mask,structure=np.ones((3,3)))
    soft = (alpha>.1)&(alpha<.9)
    band = float(soft.sum()/max(edge.sum(),1))
    outside = ~binary_dilation(mask,iterations=2)
    background = float(np.sqrt(np.mean(delta[outside]**2))) if outside.any() else 1.
    residual = delta-alpha[:,:,None]*direction
    chroma = float(np.sqrt(np.mean(residual**2)))
    core = binary_erosion(mask,iterations=2)
    if not core.any(): core=mask
    uniformity = float(np.sqrt(np.mean((image[core]-color)**2))) if core.any() else 1.
    return {'foreground_fraction':float(mask.mean()),'foreground_contrast':strength,
            'largest_component_fraction':largest,'edge_band_ratio':band,
            'background_rmse':background,'chromatic_rmse':chroma,'interior_rmse':uniformity}


def calibrate_cleanliness(images):
    scores=[cleanliness(image) for image in images]
    limits={'edge_band_ratio':max(1.2,float(np.quantile([s['edge_band_ratio'] for s in scores],.99)*1.2)),
            'background_rmse':.025,'chromatic_rmse':.025,'interior_rmse':.05,
            'min_component_fraction':.98,'min_foreground_fraction':.02,'max_foreground_fraction':.45,
            'min_contrast':.25}
    return limits


def clean_pass(scores, limits):
    return bool(limits['min_foreground_fraction']<=scores['foreground_fraction']<=limits['max_foreground_fraction']
                and scores['foreground_contrast']>=limits['min_contrast']
                and scores['largest_component_fraction']>=limits['min_component_fraction']
                and all(scores[k]<=limits[k] for k in ('edge_band_ratio','background_rmse','chromatic_rmse','interior_rmse')))
