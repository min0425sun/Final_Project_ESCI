# -*- coding: utf-8 -*-
"""
Created on 2025-10-30
boxplot side by side
62랑 



@author: Minsun
"""

#%% import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patheffects as path_effects



#%% Data path
dir_csv_62 = Path(r"D:\03.Class\ESCI\Project\01.Data\62(138)_S1A_ASCENDING_unzipped_cropped")

dir_VV_62 = Path(r"D:\03.Class\ESCI\Project\01.Data\62(138)_S1A_ASCENDING_unzipped_cropped\VV")
dir_VH_62 = Path(r"D:\03.Class\ESCI\Project\01.Data\62(138)_S1A_ASCENDING_unzipped_cropped\VH")

dir_out  = Path(r"D:\03.Class\ESCI\Project\03.Figure")
out_name = "Fig.2.boxplot.png"

#%% Load Data
#---------------------------------VH-------------------------------------------
df_VH_62 = pd.read_csv(dir_csv_62 / "extracted_backscatters_VH_path_62.csv", index_col="date", parse_dates=True)
#---------------------------------VV--------------------------------------------
df_VV_62 = pd.read_csv(dir_csv_62 / "extracted_backscatters_VV_path_62.csv", index_col="date", parse_dates=True)


df_coord_62 = pd.read_csv(dir_csv_62 / "pixel_coordinates_VH_path_62.csv")

# convert unit (dB)
df_VV_62 = df_VV_62.where(df_VV_62 > 0)   
df_VH_62 = df_VH_62.where(df_VH_62 > 0)

df_VV_62 = 10 * np.log10(df_VV_62)
df_VH_62 = 10 * np.log10(df_VH_62)

#%% 2023-24 / 2024-25
season1 = (df_VV_62.index >= "2023-10-01") & (df_VV_62.index <= "2024-05-31")   # 2023–24
season2 = (df_VV_62.index >= "2024-10-01") & (df_VV_62.index <= "2025-05-31")   # 2024–25

dates_s1 = df_VV_62.index[season1]
dates_s2 = df_VV_62.index[season2]

vv_s1 = [df_VV_62.loc[d].values.flatten() for d in dates_s1]
vh_s1 = [df_VH_62.loc[d].values.flatten() for d in dates_s1]
vv_s2 = [df_VV_62.loc[d].values.flatten() for d in dates_s2]
vh_s2 = [df_VH_62.loc[d].values.flatten() for d in dates_s2]

vv_s1 = [x[~np.isnan(x)] for x in vv_s1]
vh_s1 = [x[~np.isnan(x)] for x in vh_s1]
vv_s2 = [x[~np.isnan(x)] for x in vv_s2]
vh_s2 = [x[~np.isnan(x)] for x in vh_s2]

#%%
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharey=True)

# ---- (1) 2023–24 ----
x_s1 = np.arange(len(dates_s1))
axes[0].boxplot(
    vh_s1, positions=x_s1 + 0.15, widths=0.25,
    patch_artist=True,
    boxprops=dict(facecolor="#8FD8D2", edgecolor="black", alpha=0.8),
    medianprops=dict(color="black"),label="VH")
axes[0].boxplot(
    vv_s1, positions=x_s1 - 0.15, widths=0.25,
    patch_artist=True,
    boxprops=dict(facecolor="#FDB913", edgecolor="black", alpha=0.8),
    medianprops=dict(color="black"),label="VV")
axes[0].set_title("2023–24 Winter (Path 62, Kingman Farm)")
axes[0].set_ylabel("Backscatter (dB)")
#axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[0].legend(loc="upper right")

# ---- (2) 2024–25 ----
x_s2 = np.arange(len(dates_s2))
axes[1].boxplot(
    vh_s2, positions=x_s2 + 0.15, widths=0.25,
    patch_artist=True,
    boxprops=dict(facecolor="#8FD8D2", edgecolor="black", alpha=0.8),
    medianprops=dict(color="black"),label="VH")
axes[1].boxplot(
    vv_s2, positions=x_s2 - 0.15, widths=0.25,
    patch_artist=True,
    boxprops=dict(facecolor="#FDB913", edgecolor="black", alpha=0.8),
    medianprops=dict(color="black"),label="VV")
axes[1].set_title("2024–25 Winter (Path 62, Kingman Farm)")
axes[1].set_ylabel("Backscatter (dB)")
axes[1].set_xlabel("Date")
#axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[1].legend(loc="upper right")

# ---- x축 라벨 (날짜) ----
for ax, dates in zip(axes, [dates_s1, dates_s2]):
    ax.set_xticks(np.arange(len(dates))[::2])
    ax.set_xticklabels(
        [d.strftime("%Y-%m-%d") for d in dates[::2]],
        rotation=45, ha="right"
    )

plt.tight_layout()
plt.savefig(dir_out / "Fig2_boxplot_by_two_winter.png", dpi=300, bbox_inches="tight")
plt.show()


#%%


























