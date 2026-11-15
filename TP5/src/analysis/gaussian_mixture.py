"""Mezcla gaussiana de covarianza completa ajustada por EM, solo NumPy/SciPy.

Es una densidad auxiliar posterior al entrenamiento; no es parte de la pérdida
del AE/VAE. Los reinicios se eligen por verosimilitud de entrenamiento.
"""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp


def component_logpdf(samples,means,covariances):
    columns=[]
    for mean,covariance in zip(means,covariances):
        root=np.linalg.cholesky(covariance)
        normalized=solve_triangular(root,(samples-mean).T,lower=True,check_finite=False)
        columns.append(-.5*(samples.shape[1]*np.log(2*np.pi)
                       +2*np.log(np.diag(root)).sum()+(normalized**2).sum(axis=0)))
    return np.column_stack(columns)


def initial_means(samples,count,rng):
    chosen=[samples[rng.integers(len(samples))].copy()]
    distances=np.sum((samples-chosen[0])**2,axis=1)
    for _ in range(1,count):
        total=distances.sum()
        index=rng.choice(len(samples),p=distances/total) if total>0 else rng.integers(len(samples))
        chosen.append(samples[index].copy())
        distances=np.minimum(distances,np.sum((samples-chosen[-1])**2,axis=1))
    return np.asarray(chosen)


def fit_mixture(samples,components,seed=0,restarts=3,max_iterations=100,tolerance=1e-5,ridge=1e-5):
    samples=np.asarray(samples,dtype=np.float64)
    if (samples.ndim!=2 or not np.isfinite(samples).all() or len(samples)<2
            or not 1<=components<=len(samples) or restarts<1 or max_iterations<1 or ridge<=0):
        raise ValueError('Datos o parámetros inválidos para la mezcla')
    rng=np.random.default_rng(seed)
    centered=samples-samples.mean(axis=0)
    initial_cov=centered.T@centered/len(samples)+ridge*np.eye(samples.shape[1])
    best=None
    for _ in range(restarts):
        means=initial_means(samples,components,rng)
        covariances=np.repeat(initial_cov[None],components,axis=0)
        weights=np.full(components,1/components)
        history=[]
        for _ in range(max_iterations):
            scores=component_logpdf(samples,means,covariances)+np.log(weights)
            normalization=logsumexp(scores,axis=1)
            likelihood=float(normalization.mean());history.append(likelihood)
            if len(history)>1 and abs(history[-1]-history[-2])<tolerance:break
            responsibilities=np.exp(scores-normalization[:,None])
            sizes=responsibilities.sum(axis=0)
            if np.any(sizes<1e-8):
                # Reinitialize a degenerate component from training, never validation.
                for k in np.flatnonzero(sizes<1e-8):
                    responsibilities[:,k]=1e-4
                responsibilities/=responsibilities.sum(axis=1,keepdims=True)
                sizes=responsibilities.sum(axis=0)
            weights=sizes/len(samples)
            means=(responsibilities.T@samples)/sizes[:,None]
            for k in range(components):
                offsets=samples-means[k]
                covariances[k]=(offsets.T*responsibilities[:,k])@offsets/sizes[k]
                covariances[k]+=ridge*np.eye(samples.shape[1])
        final=float(logsumexp(component_logpdf(samples,means,covariances)+np.log(weights),axis=1).mean())
        result={'weights':weights.copy(),'means':means.copy(),'covariances':covariances.copy(),
                'train_log_likelihood':final,'iterations':len(history),'history':history}
        if best is None or final>best['train_log_likelihood']:best=result
    return best


def sample_mixture(fitted,epsilon,uniforms):
    epsilon=np.asarray(epsilon,dtype=np.float64);uniforms=np.asarray(uniforms,dtype=np.float64)
    if (epsilon.ndim!=2 or epsilon.shape[1]!=fitted['means'].shape[1]
            or uniforms.shape!=(len(epsilon),) or not np.isfinite(epsilon).all()
            or not np.isfinite(uniforms).all() or np.any(uniforms<0) or np.any(uniforms>=1)):
        raise ValueError('Ruido normal o uniforme inválido')
    bounds=np.cumsum(fitted['weights']);bounds[-1]=1.
    components=np.searchsorted(bounds,uniforms,side='right')
    coordinates=np.empty_like(epsilon)
    for k in range(len(bounds)):
        chosen=components==k
        root=np.linalg.cholesky(fitted['covariances'][k])
        coordinates[chosen]=epsilon[chosen]@root.T+fitted['means'][k]
    return coordinates,components
