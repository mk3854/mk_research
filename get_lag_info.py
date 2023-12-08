#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from numpy.random import randn
from scipy.stats import norm, skewnorm
from shapedtw.shapedtw import shape_dtw
from shapedtw.shapeDescriptors import SlopeDescriptor, PAADescriptor, DerivativeShapeDescriptor, CompoundDescriptor
from shapedtw.dtwPlot import dtwPlot
from shapedtw.shapedtw import DistanceReconstructor


# In[3]:


def make_path(spl, case):
    slope_descriptor = SlopeDescriptor(slope_window=2)
    paa_descriptor = PAADescriptor(piecewise_aggregation_window=3)
    derivs_descriptor=DerivativeShapeDescriptor()

    # compound_descriptor = CompoundDescriptor([slope_descriptor, derivs_descriptor],descriptors_weights=[5., 1.])
    compound_descriptor = CompoundDescriptor([slope_descriptor, paa_descriptor, derivs_descriptor],descriptors_weights=[5., 1., 1.])

    shape_dtw_results = shape_dtw(
        x=spl.to_numpy(),
        y=case.to_numpy(),
        subsequence_width=[3,4,7,8,9,10,20][2],
        shape_descriptor=[slope_descriptor, paa_descriptor, derivs_descriptor,compound_descriptor][-1],
        window_type="sakoechiba",
        window_args={"window_size": 4}, #最大ラグ
        keep_internals=True
    )
    
    dist_reconstructor = DistanceReconstructor(
    step_pattern="symmetric2",
    ts_x=spl.to_numpy(),
    ts_y=case.to_numpy(),
    ts_x_wp=shape_dtw_results.index1s,
    ts_y_wp=shape_dtw_results.index2s,
    dist_method="euclidean"
)
    path = dist_reconstructor._get_indices_pairs()
    return path


def make_lag(path, case):
    path_c =  [[] for i in range(len(case.to_numpy()))]
    for k in path:
        path_c[k[1]].append(k[0])
    dist = [np.mean(k) for k in path_c]

    dist_c = [n-k for n,k in enumerate(dist)]
    return dist_c


# In[ ]:


def plot_shape_dtw(spl, case, label,label2="Case", offset=0.4, lag_series=False):
    slope_descriptor = SlopeDescriptor(slope_window=2)
    paa_descriptor = PAADescriptor(piecewise_aggregation_window=3)
    derivs_descriptor=DerivativeShapeDescriptor()
    # compound_descriptor = CompoundDescriptor([slope_descriptor, derivs_descriptor],descriptors_weights=[5., 1.])
    compound_descriptor = CompoundDescriptor([slope_descriptor, paa_descriptor, derivs_descriptor],descriptors_weights=[5., 1., 1.])

    res = shape_dtw(
        x=spl.to_numpy(),
        y=case.to_numpy(),
        subsequence_width=[3,4,7,8,9,10,20][2],
        shape_descriptor=[slope_descriptor, paa_descriptor, derivs_descriptor,compound_descriptor][-1],
        window_type="sakoechiba",
        window_args={"window_size": 4}, #最大ラグ
        keep_internals=True
    )

    plt.figure(figsize=(12,6))
    for a,b in zip(res.index1s, res.index2s):
        plt.plot([a,b], [spl.to_numpy()[a]+offset, case.to_numpy()[b]], "--r")
    plt.plot(spl.to_numpy()+offset, "-ob", ms=3, label=label)
    plt.plot(case.to_numpy(), "-ok", ms=3, label="Case")
    plt.legend()

    if lag_series:
        plt.figure(figsize=(12,6))
        dist_c=make_lag([(i,k) for i,k in zip(res.index1s, res.index2s)], case)
        dist_c=[math.ceil(i) for i in dist_c]
        plt.plot(dist_c, "-o", ms=2)
        plt.hlines(0, 0, 140, colors="gray", linestyles="dotted")
        plt.text(70, min(dist_c)/2, "下水遅れ")
        plt.text(70, max(dist_c)/2, "下水早い")


# In[ ]:


def make_lag_df(dist, case):
    return pd.get_dummies(pd.DataFrame(dist, index=case.index, columns=["lag"]), columns=["lag"],dtype=int)


# In[ ]:


def make_spl_lag(spl, case):
    path = make_path(spl, case)

    dist = make_lag(path, case)

    dist = [math.ceil(i) for i in dist]
    lag_df = make_lag_df(dist, case)

    col=np.arange(-4, 7, 1)[np.arange(-4, 7) < min(dist)]
    lag_df=pd.concat([pd.DataFrame(0,index=case.index, columns=[f"lag_{i}"for i in col], dtype=int), lag_df], axis=1)
    col=np.arange(-4, 7, 1)[np.arange(-4, 7) > max(dist)]
    lag_df=pd.concat([lag_df, pd.DataFrame(0,index=case.index, columns=[f"lag_{i}"for i in col], dtype=int)], axis=1)

    lag_df = lag_df.iloc[:,::-1]

    return lag_df


# In[ ]:


def lag_norm(lags, s2=1.75, ymin=0):
    x=range(len(lags))
    y=norm.pdf(x,lags.index(max(lags)), s2)+ymin
    return y


# In[ ]:


def norm_lag_df(df, s2=1.75, ymin=-0.05):
    norm_lag = pd.DataFrame(np.array([lag_norm(n.to_list(), s2, ymin) for _,n in df.iterrows()])[:,:-5])
    norm_lag.index=df.index
    norm_lag.columns=df.columns[:-5]
    return norm_lag

