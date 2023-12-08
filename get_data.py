#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
import math

from scipy.interpolate import splrep, splev
from scipy.interpolate import BSpline, CubicSpline, make_smoothing_spline
from scipy.interpolate import interp1d
from scipy.stats import norm, skewnorm
from datetime import timedelta

def get_data():
    #感染者数をインポート
    df = pd.read_csv('data/Daily COVID-19 reported cases.csv', skiprows=0, index_col='Date', parse_dates=True)
    dataset=df.resample("W").sum()[["NumberCasesPerDay"]]
    dataset=dataset.rename(columns={"NumberCasesPerDay":"Case"})
    #下水データをインポート
    df_RNA = pd.read_table('data/RNAMonitoring_Public - Result Description - N1 Gene, Reported Value - N1 Gene (gc-l), Days Since.csv', skiprows=0, index_col="Date", parse_dates=True, encoding="UTF-16")
    df_RNA=df_RNA[df_RNA["Health Area"]=="Greater Glasgow and Clyde"]
    #必要な地域だけ残す
    df_RNA2=df_RNA[df_RNA["Site Name"]=="Dalmuir"]
    df_RNA2=df_RNA2[["Reported Value - N1 Gene (gc/l)"]]
    df_RNA=df_RNA[df_RNA["Site Name"]=="Shieldhall"]
    df_RNA=df_RNA[["Reported Value - N1 Gene (gc/l)"]]
    #下水データを週単位にする
    data_RNA=df_RNA.resample("W").mean()
    data_RNA=data_RNA.interpolate()
    data_RNA.columns = ["Conc_S"]

    data_RNA2=df_RNA2.resample("W").mean()
    data_RNA2=data_RNA2.interpolate()
    data_RNA2.columns = ["Conc_D"]
    #スプライン補間のための準備
    x = np.array((df_RNA.index-pd.to_datetime('2020-05-28'))/ timedelta(days=1))
    y = np.array(df_RNA["Reported Value - N1 Gene (gc/l)"], dtype=np.float64)

    x2 = (pd.date_range(start=df_RNA.index[0],freq='d',end=df_RNA.index[-1])-pd.to_datetime('2020-05-28'))/ timedelta(days=1)
    x2=np.array(x2)
    x3=pd.date_range(start=df_RNA.index[0],freq='d',end=df_RNA.index[-1])

    spl = make_smoothing_spline(x, y, lam=10**3) #スプライン補完の強度

    spl_data=pd.DataFrame(pd.Series(spl(x2), index=pd.date_range(start=df_RNA.index[0],freq='d',end=df_RNA.index[-1])))
    (spl_data)[spl_data<0]=0

    x = np.array((df_RNA2.index-pd.to_datetime('2020-05-29'))/ timedelta(days=1))
    y = np.array(df_RNA2["Reported Value - N1 Gene (gc/l)"], dtype=np.float64)

    x2 = (pd.date_range(start=df_RNA2.index[0],freq='d',end=df_RNA2.index[-1])-pd.to_datetime('2020-05-29'))/ timedelta(days=1)
    x2=np.array(x2)
    x3=pd.date_range(start=df_RNA2.index[0],freq='d',end=df_RNA2.index[-1])

    spl = make_smoothing_spline(x, y, lam=10**3) #スプライン補完の強度

    spl_data2=pd.DataFrame(pd.Series(spl(x2), index=pd.date_range(start=df_RNA2.index[0],freq='d',end=df_RNA2.index[-1])))
    (spl_data2)[spl_data2<0]=0

    #data_setRNA : 下水のコロナウイルス量
    data_RNA["Conc_D"]=data_RNA2["Conc_D"]
    data_RNA=np.log(data_RNA+1)

    data_RNA["Case"]=np.log(dataset["2020-05-31":]["Case"]+1)
    data_RNA=(data_RNA)

    spl_dataset=np.log(spl_data.resample("W").mean()+1)
    data_RNA["spl_S"]=spl_dataset

    spl_dataset2=np.log(spl_data2.resample("W").mean()+1)
    data_RNA["spl_D"]=spl_dataset2

    data_RNA["spl_S&D"]=(data_RNA["spl_S"]+data_RNA["spl_D"])/2

    data_RNA=data_RNA[12:]
    data_RNA=data_RNA[["Case", "Conc_S", "Conc_D", "spl_S", "spl_D", "spl_S&D"]] #感染者数、ウイルス濃度S、ウイルス濃度D、ウイルス濃度(補間済み)S、ウイルス濃度(補間済み)D、ウイルス濃度平均(補間済み)
    
    return data_RNA



