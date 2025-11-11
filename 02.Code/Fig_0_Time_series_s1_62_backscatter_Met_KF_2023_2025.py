# -*- coding: utf-8 -*-
"""
data input: KF CRN data, sentinel-1 backscatter.

Temporal evolution of Sentinel-1 ascending (Path 62) VV and VH polarized
backscatter coefficients (dB) and in-situ meteorological variables form CRN
(air temperature and precipitation) at Kingman Farm

@License: MIT
@author: Minsun
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

#%% Load S1 data
dir_csv_62 = Path(r"D:\03.Class\ESCI\Project\01.Data\62(138)_S1A_ASCENDING_unzipped_cropped")

dir_VV_62 = Path(r"D:\03.Class\ESCI\Project\01.Data\62(138)_S1A_ASCENDING_unzipped_cropped\VV")
dir_VH_62 = Path(r"D:\03.Class\ESCI\Project\01.Data\62(138)_S1A_ASCENDING_unzipped_cropped\VH")

#---------------------------------VH-------------------------------------------
df_VH_62 = pd.read_csv(dir_csv_62 / "extracted_backscatters_VH_path_62.csv", index_col="date", parse_dates=True)
#---------------------------------VV--------------------------------------------
df_VV_62 = pd.read_csv(dir_csv_62 / "extracted_backscatters_VV_path_62.csv", index_col="date", parse_dates=True)

# convert unit (dB)
df_VV_62 = df_VV_62.where(df_VV_62 > 0)   
df_VH_62 = df_VH_62.where(df_VH_62 > 0)

df_VV_62 = 10 * np.log10(df_VV_62)
df_VH_62 = 10 * np.log10(df_VH_62)

# two different winter
season1 = (df_VV_62.index >= "2023-10-01") & (df_VV_62.index <= "2024-05-31")   # 2023–24
season2 = (df_VV_62.index >= "2024-10-01") & (df_VV_62.index <= "2025-05-31")   # 2024–25

mean_VV_s1 = df_VV_62.loc[season1].mean(axis=1)
mean_VH_s1 = df_VH_62.loc[season1].mean(axis=1)
mean_VV_s2 = df_VV_62.loc[season2].mean(axis=1)
mean_VH_s2 = df_VH_62.loc[season2].mean(axis=1)

mean_backscatter = pd.DataFrame({
    "VV_2023_24": mean_VV_s1,
    "VH_2023_24": mean_VH_s1,
    "VV_2024_25": mean_VV_s2,
    "VH_2024_25": mean_VH_s2
})

#%% Load Met data
dir_csv_met=Path(r'D:\03.Class\ESCI\Project\01.Data\met_data')
files = [f for f in os.listdir(dir_csv_met) if f.endswith('.xlsx') and 'CRN_Kingman' in f]

df_list = []
for f in files:
    file_path = os.path.join(dir_csv_met, f)
    temp = pd.read_excel(file_path,skiprows=1)
    df_list.append(temp)
Met = pd.concat(df_list, ignore_index=True)
Met = Met[Met['LST_DATE'] != 'YYYYMMDD'] #remove str

Met['LST_DATE'] = pd.to_datetime(Met['LST_DATE'], format='%Y%m%d')
Met.set_index('LST_DATE', inplace=True)
Met = Met[['T_DAILY_MEAN', 'SOLARAD_DAILY','P_DAILY_CALC']]
Met.replace(-9999, np.nan, inplace=True)


#%% Clac average
t_mean = Met['T_DAILY_MEAN'].mean()
s_mean = Met['SOLARAD_DAILY'].mean()
p_mean = Met['P_DAILY_CALC'].mean()

print(f"Winter mean temperature: {t_mean:.2f} °C")
print(f"Winter mean solar radiation: {s_mean:.2f} MJ/m²/day")
print(f"Winter mean precipitation: {p_mean:.2f} mm/day")

#%%
start, end = "2023-10-01", "2025-05-31"
Met = Met.loc[start:end]
mean_backscatter = mean_backscatter.loc[start:end]

fig, ax1 = plt.subplots(figsize=(20,5))

# (1) Backscatter (왼쪽 y축)
ax1.plot(mean_backscatter.index, mean_backscatter["VV_2023_24"], 
         "o-", color="#FDB913")
ax1.plot(mean_backscatter.index, mean_backscatter["VH_2023_24"], 
         "o-", color="#8FD8D2", label="VH")
ax1.plot(mean_backscatter.index, mean_backscatter["VV_2024_25"], 
         "o-", color="#FDB913", label="VV")
ax1.plot(mean_backscatter.index, mean_backscatter["VH_2024_25"], 
         "o-", color="#8FD8D2")

ax1.set_ylabel("Backscatter [dB]")
ax1.tick_params(axis="y", labelcolor="black")

# (2) Air temperature (right)
ax2 = ax1.twinx()
ax2.plot(Met.index, Met["T_DAILY_MEAN"], color="red",alpha=0.4,linewidth=1, label="Air Temp (°C)")
ax2.axhline(0, color="red", linestyle="--", linewidth=0.8)
ax2.set_ylabel("Air Temperature [°C]", color="red")
ax2.tick_params(axis="y", labelcolor="red")

# (3) Precipitation (bar, right outside)
ax3 = ax1.twinx()
ax3.bar(Met.index, Met["P_DAILY_CALC"], color="cornflowerblue", alpha=0.4, width=2, label="Precip (mm)")
ax3.spines["right"].set_position(("outward", 50))
ax3.set_ylabel("Precipitation [mm]", color="blue")

# (4) lagend
lines, labels = [], []
for ax in [ax1, ax2, ax3]:
    l, lb = ax.get_legend_handles_labels()
    lines += l; labels += lb
ax1.legend(lines, labels, loc="upper right", frameon=False)


ax1.set_title("Sentinel-1 and Meteological Condition(Kingman Farm)")
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
