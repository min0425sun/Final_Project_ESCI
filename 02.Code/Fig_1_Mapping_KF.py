# -*- coding: utf-8 -*-
"""

@author: Minsun
"""

#%% import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import show
from matplotlib.patches import Rectangle
from pathlib import Path
import matplotlib.patheffects as path_effects


#%% Path
dir_csv_62 = Path(r"D:\03.Class\ESCI\Project\01.Data\62(138)_S1A_ASCENDING_unzipped_cropped")


dir_VV_62 = Path(r"D:\03.Class\ESCI\Project\01.Data\62(138)_S1A_ASCENDING_unzipped_cropped\VV")
dir_VH_62 = Path(r"D:\03.Class\ESCI\Project\01.Data\62(138)_S1A_ASCENDING_unzipped_cropped\VH")

# Shapefile load
boundary_file = Path(r"D:\03.Class\ESCI\Project\01.Data\Kingman_Farm_boundary\Kingman_Farm_boundary\Kingman_Farm_boundary.shp")


#%% individual image out directory
#dir_out  = Path(r"D:\03.Class\ESCI\Project\03.Figure\Mapping_VV_62") #
dir_out  = Path(r"D:\03.Class\ESCI\Project\03.Figure\Mapping_VH_62") #

#basemap_file = dir_basemap / "Basemap_32619(62).tif"



#vv_files = list(dir_VV_62.glob("*.tif"))  # 현재 궤도 폴더에 있는 모든 VV tif 불러오기
vh_files = list(dir_VH_62.glob("*.tif"))  # 현재 궤도 폴더에 있는 모든 VH tif 불러오기

#%%
boundary = gpd.read_file(boundary_file).to_crs(epsg=32619)

boundary_style = dict(
    edgecolor='black',
    facecolor='none',
    linewidth=1.5,
    zorder=5
)



# ===============================
# Figure
n = len(vh_files) #change VV or VH file
cols = 7
rows = int(np.ceil(n / cols))
figsize_per_tile = 3.0
fig_all, axes = plt.subplots(rows, cols,
                             figsize=(cols * figsize_per_tile, rows * figsize_per_tile),
                             constrained_layout=True)
axes = axes.flatten()
cmap = plt.cm.Greys_r #color code

vmin, vmax = np.inf, -np.inf
for i, f in enumerate(vh_files): # call satellite image (.tif)
    fname = f.stem #file name (string)
    date_str = fname.split('_')[2][:8] #3rd position and read only date ex)"20231215"

    with rasterio.open(f) as src: #open GeoTIFF / with: close file automatically to save memory
        sar = src.read(1).astype("float64") # read 1st band
        sar[sar <= 0] = np.nan
        sar_dB = 10 * np.log10(sar) # convert unit (desibel)
        extent = [
            src.transform[2], #left (x_min)
            src.transform[2] + src.transform[0] * src.width,# right (x_max)
            src.transform[5] + src.transform[4] * src.height, # bottom (y_min)
            src.transform[5], # top (y_max)
        ]
    vmin = min(vmin, np.nanmin(sar_dB))
    vmax = max(vmax, np.nanmax(sar_dB))
    ax = axes[i]
    im = ax.imshow(sar_dB, extent=extent, origin='upper',
                   cmap=cmap, vmin=vmin, vmax=vmax)

    # Kingman boundary (outer)
    boundary.plot(ax=ax, **boundary_style)


    # Date label
    ax.text(0.65, 0.05, date_str,
            color="firebrick", fontsize=9, fontweight='bold',
            transform=ax.transAxes,
            )
    ax.set_xticks([]); ax.set_yticks([])


for ax in axes[len(vh_files):]:
    ax.axis('off')
    
cbar = fig_all.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
cbar.set_label("Backscatter (dB)", fontsize=11)

#%%
# =============================================================== Code End  ====================================================================




#%% supplementary; individual image
# calc dB range
vmin, vmax = np.inf, -np.inf

for f in vv_files:
    with rasterio.open(f) as src:
        arr = src.read(1).astype("float64")
        arr[arr <= 0] = np.nan
        arr_dB = 10 * np.log10(arr)
        vmin = min(vmin, np.nanmin(arr_dB))
        vmax = max(vmax, np.nanmax(arr_dB))

# 
vmin = np.floor(vmin * 10) / 10    # 아래쪽으로 반올림
vmax = np.ceil(vmax * 10) / 10     # 위쪽으로 반올림

print(f" dB range: {vmin:.1f} ~ {vmax:.1f}")


#%% 
for f in vv_files:
    fname = f.stem                          # 전체 파일명
    parts = fname.split('_')                # 언더바 기준으로 분리
    date_str = parts[2][:8]                 # 예: 20250520
    pol = "VV" if "VV" in fname else "VH"   # 자동 편파 인식

    print(f"Processing {date_str} ({pol}) ...")
    with rasterio.open(f) as src:
        arr = src.read(1).astype("float64")
        arr[arr <= 0] = np.nan
        arr_dB = 10 * np.log10(arr)
        extent = [
            src.transform[2],
            src.transform[2] + src.transform[0] * src.width,
            src.transform[5] + src.transform[4] * src.height,
            src.transform[5],
        ]

    fig, ax = plt.subplots(figsize=(9, 9))
    #show(basemap, transform=basemap_transform, ax=ax)

    # 
    im = ax.imshow(arr_dB, extent=extent, origin='upper',
                   cmap='gray', vmin=vmin, vmax=vmax)

    #ax.set_xlim(basemap_xlim)
    #ax.set_ylim(basemap_ylim)

    cbar = plt.colorbar(im, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label("Backscatter (dB)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(fname, fontsize=11)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    out_name = dir_out / f"{date_str}_{pol}.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  → saved: {out_name.name}")














