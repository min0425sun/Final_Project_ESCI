import os
import re
import glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import pyproj
from pyproj import Transformer


## Settings
# 1. GeoTiff
data_dir = Path(".")
vv_dir = data_dir / "Sentinel-1" / "VV"
vh_dir = data_dir / "Sentinel-1" / "VH"

# 2. Backscatter value (csv)
df_VH_62 = pd.read_csv(data_dir / "Sentinel-1" / "extracted_backscatters_VH_path_62.csv", index_col="date", parse_dates=True)
df_VV_62 = pd.read_csv(data_dir / "Sentinel-1" / "extracted_backscatters_VV_path_62.csv", index_col="date", parse_dates=True)

# Convert to dB
df_VV_62 = 10 * np.log10(df_VV_62.where(df_VV_62 > 0))
df_VH_62 = 10 * np.log10(df_VH_62.where(df_VH_62 > 0))

# 3. Hubbard Brook lat, lon
# (EPSG:4326 → EPSG:32619)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True)

def latlon_to_utm(lon, lat):
    x, y = transformer.transform(lon, lat)
    return x, y

lat = 43 + 57/60
lon = -(71 + 44/60)

hb_x, hb_y = latlon_to_utm(lon, lat)
hb_x, hb_y


## In situ data
Insitu = pd.read_csv(data_dir / "in_situ_data" / "Insitu_Snow_Met_HB_2023_2025.csv",skiprows=2)
Insitu['Date'] = pd.to_datetime(Insitu['Date'])

# in → cm (1 inch = 2.54 cm)
Insitu['snow_depth_cm'] = Insitu['SNWD.I-1 (in) '] * 2.54

Insitu.head()


## Function to extract sentinel-1 GeoTiff pixel align with in situ data
# 1) Extract pixel value from SAR file
def extract_pixel_value(tiff_path, x, y):
    try:
        with rasterio.open(tiff_path) as src:
            row, col = src.index(x, y)  # ← UTM Easting, Northing
            pixel = src.read(1)[row, col]
        return pixel
    except:
        return np.nan    
# ----------------------------------------------------------

# 2) Extract date from filename
# Example: S1A_IW_20231010T223545_DVP_RTC30_G_gpufem_09B1_VV_62.tif

def extract_date(fname):
    # divide filename by underscore (_)
    parts = fname.split("_") #['S1A', 'IW', '20231010T223545', 'DVP', 'RTC30', 'G', 'gpufem', '09B1', 'VV', '62.tif']

    # parts[2] = '20231010T223545'
    timestamp = parts[2]
    
    # '20231010T223545' → datetime
    dt = datetime.strptime(timestamp, "%Y%m%dT%H%M%S")
    
    return dt.date()
# ----------------------------------------------------------

# 3) Load SAR folder
def load_sar_folder(folder, pol):
    sar_records = []
    for file in sorted(folder.glob("*.tif")): #file = .tif file path ex) VV\S1A_IW_20231010T223545_DVP_RTC30_G_gpufem_09B1_VV_62.tif
        fname = file.name
        d = extract_date(fname)
        if d is None:
            continue
        val = extract_pixel_value(file, hb_x, hb_y)

        if val is not None and val > 0:
            val_db=10 * np.log10(val)
        else:
            val_db=np.nan
        sar_records.append([d, pol, val_db])
    return pd.DataFrame(sar_records, columns=["Date", "Pol", "Value"])


## Load SAR
vv_df = load_sar_folder(vv_dir, "VV")
vh_df = load_sar_folder(vh_dir, "VH")

sar_df = pd.concat([vv_df, vh_df]).pivot(index="Date", columns="Pol", values="Value")
sar_df.index = pd.to_datetime(sar_df.index)
sar_df = sar_df.sort_index()

sar_df.head()


## Merge data with in situ observation
merged = Insitu.merge(sar_df, on="Date", how="left")
merged['Year'] = merged['Date'].dt.year
merged.head(10)


# %%
## Calculation: delta / Stage (Snow-free, accumulation, melt)
df = merged.copy()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 1) Delta snow depth
df['dSD'] = df['snow_depth_cm'].diff()

# 2-1) Stage - Snow-free
df['Stage'] = 'Snow-free'

# 2-2) Stage - Accumulation
df.loc[df['dSD'] > 0, 'Stage'] = 'Accumulation'

# 2-3) Stage - Melt: reduction in snow
df.loc[df['dSD'] < 0, 'Stage'] = 'Melt'

# 3) 'Snow-free' when snow exists → Accumulation
df.loc[(df['snow_depth_cm'] > 0) & (df['Stage'] == 'Snow-free'), 'Stage'] = 'Accumulation'

df.head(20)


# %%
## Snow time series
fig, axes = plt.subplots(2, 1, figsize=(13, 10))

# ---------------------------------------------------------
# (a) Snow depth + Temperature (dual axis)
# ---------------------------------------------------------
ax1 = axes[0]       # left axis
ax2 = ax1.twinx()   # right axis

# temperature
line_temp, = ax2.plot(df['Date'], df['TOBS.I-1 (degC) '],
                     color='#1f77b4', lw=1, label='Temperature (°C)')

ax2.axhline(0, color='#1f77b4', linestyle='--', linewidth=1)
ax2.set_ylim(-20, 30)

# snow depth
line_sd, = ax1.plot(df['Date'], df['snow_depth_cm'], 'k-', lw = 2, label = 'Snow depth (cm)')
ax1.set_ylim(0, 100)
ax1.set_xlim(datetime(2023, 10, 1), datetime(2025, 5, 30))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))

# shading
acc = df['Stage'] == 'Accumulation'
melt = df['Stage'] == 'Melt'

ymax = ax1.get_ylim()[1]

ax1.fill_between(df['Date'], 0, ymax,
                 where=acc, color='lightblue', alpha=0.25, label = 'Accumulation')

ax1.fill_between(df['Date'], 0, ymax,
                 where=melt, color='pink', alpha=0.25, label = 'Melt')

# label
ax1.set_ylabel("Snow depth (cm)")
ax2.set_ylabel("Temperature (°C)")

# legend
ax1.legend(handles=[line_sd, line_temp])

# title
ax1.set_title("(a) Snow depth with shading (blue = accumulation, pink = melt) and Air temperature", loc = 'left', fontsize = 14)


# ---------------------------------------------------------
# (b) Snow depth + backscatter
# ---------------------------------------------------------
ax1 = axes[1]
ax2 = ax1.twinx()

# snow depth
line_sd, = ax1.plot(df['Date'], df['snow_depth_cm'], 'k-', lw = 2, label = 'Snow depth (cm)')
ax1.set_ylim(0, 100)
ax1.set_xlim(datetime(2023, 10, 1), datetime(2025, 5, 30))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))

# shading
acc = df['Stage'] == 'Accumulation'
melt = df['Stage'] == 'Melt'

ymax = ax1.get_ylim()[1]

ax1.fill_between(df['Date'], 0, ymax,
                 where=acc, color='lightblue', alpha=0.25, label = 'Accumulation')

ax1.fill_between(df['Date'], 0, ymax,
                 where=melt, color='pink', alpha=0.25, label = 'Melt')


# backscatter
season1 = (df['Date'] >= datetime(2023,10,1)) & (df['Date'] <= datetime(2024,5,31))
season2 = (df['Date'] >= datetime(2024,10,1)) & (df['Date'] <= datetime(2025,5,31))

# VV
line_VV, = ax2.plot(df.loc[season1 & df['VV'].notna(), 'Date'],
          df.loc[season1 & df['VV'].notna(), 'VV'],
          'o-', color="#FDB913", label="VV backscatter (dB)")

ax2.plot(df.loc[season2 & df['VV'].notna(), 'Date'],
          df.loc[season2 & df['VV'].notna(), 'VV'],
          'o-', color="#FDB913", label="VV backscatter")

# VH
line_VH, = ax2.plot(df.loc[season1 & df['VH'].notna(), 'Date'],
          df.loc[season1 & df['VH'].notna(), 'VH'],
          'o-', color="#8FD8D2", label="VH backscatter (dB)")

ax2.plot(df.loc[season2 & df['VH'].notna(), 'Date'],
          df.loc[season2 & df['VH'].notna(), 'VH'],
          'o-', color="#8FD8D2", label="VH backscatter")

ax2.set_ylim(-17, 0)

# label
ax1.set_ylabel("Snow depth (cm)")
ax2.set_ylabel("Backscatter (dB)")

# legend
ax1.legend(handles=[line_sd, line_VV, line_VH])

# title
ax1.set_title("(b) Snow depth with shading (blue = accumulation, pink = melt) and SAR Backscatter", loc = 'left', fontsize = 14)


plt.show()


# %%
# Define winter periods
season1 = (df_VV_62.index >= "2023-10-01") & (df_VV_62.index <= "2024-05-31")
season2 = (df_VV_62.index >= "2024-10-01") & (df_VV_62.index <= "2025-05-31")

dates_s1 = df_VV_62.index[season1]
dates_s2 = df_VV_62.index[season2]

vv_s1 = [df_VV_62.loc[d].dropna().values for d in dates_s1]
vh_s1 = [df_VH_62.loc[d].dropna().values for d in dates_s1]
vv_s2 = [df_VV_62.loc[d].dropna().values for d in dates_s2]
vh_s2 = [df_VH_62.loc[d].dropna().values for d in dates_s2]

# figure
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)

titles = [
    "(a) 2023–24 Winter — VV (Hubbard Brook)",
    "(b) 2023–24 Winter — VH (Hubbard Brook)",
    "(c) 2024–25 Winter — VV (Hubbard Brook)",
    "(d) 2024–25 Winter — VH (Hubbard Brook)"
]
colors = ["#FDB913", "#8FD8D2", "#FDB913", "#8FD8D2"]

data_list = [vv_s1, vh_s1, vv_s2, vh_s2]
dates_list = [dates_s1, dates_s1, dates_s2, dates_s2]

# ---------------------------------------------------------
# Helper: Boxplot function
# ---------------------------------------------------------
def plot_boxes(ax, data, dates, title, color):
    x_pos = np.arange(len(dates))

    ax.boxplot(
        data,
        positions=x_pos,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor=color, edgecolor="black", alpha=0.8),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black")
    )

    ax.set_title(title, fontsize=14, loc='left')
    ax.set_ylabel("Backscatter (dB)")

    ax.set_xticks(x_pos[::2])
    ax.set_xticklabels([d.strftime("%d %b") for d in dates[::2]])

for ax, data, dates, title, color in zip(
        axes.flatten(), data_list, dates_list, titles, colors):
    plot_boxes(ax, data, dates, title, color)

plt.tight_layout()
plt.show()


# %% Map
# Select polarization (VH)
polarization = "VH" # 
data_dir = vh_dir if polarization == "VH" else vv_dir

# Load file
files = sorted(list(data_dir.glob("*.tif"))) #list of GeoTIFF files
files = sorted(data_dir.glob("*.tif"))
if not files:
    raise FileNotFoundError(f"No .tif files ")
print(f"{len(files)} {polarization} images")

# Select polarization (VV)

polarization2 = "VV" # 
data_dir = vv_dir if polarization2 == "VV" else vh_dir

# Load file
files2 = sorted(list(data_dir.glob("*.tif"))) #list of GeoTIFF files
files2 = sorted(data_dir.glob("*.tif"))
if not files2:
    raise FileNotFoundError(f"No .tif files ")
print(f"{len(files2)} {polarization2} images")


# %%
# Mapping sentinel-1 

n = len(files)
cols = 7
rows = int(np.ceil(n / cols))
figsize_per_tile = 1.8
fig_all, axes = plt.subplots(rows, cols,
                             figsize=(cols * figsize_per_tile, rows * figsize_per_tile),
                             constrained_layout=True)
axes = axes.flatten()

#initialize value to keep the minimum and maximum values across all images.
#Since dB values include negative numbers, set +/- infinite.
vmin, vmax = np.inf, -np.inf 

for i, f in enumerate(files): # give both the index (i) and the file path (f)
    with rasterio.open(f) as src: #open GeoTIFF using the rasterio library and automatically closes the file after reading
        sar = src.read(1).astype(float) # read 1st band and converts it to float (floats are needed for log10)
        sar[sar <= 0] = np.nan # 'sar' contains backscatter values
        sar_dB = 10 * np.log10(sar) # convert unit (desibel)
        extent = [ # defines the geographic information of the image (map coordinates).
            src.transform[2], #left (x_min)
            src.transform[2] + src.transform[0] * src.width, # move width * pixel size step to right (x_max) 
            src.transform[5] + src.transform[4] * src.height, # move height * pixel size step to bottom down (y_min)
            src.transform[5],# top (y_max)
        ]
    vmin = min(vmin, np.nanmin(sar_dB))
    vmax = max(vmax, np.nanmax(sar_dB))
    ax = axes[i]
    im = ax.imshow(sar_dB, extent=extent, origin="upper", cmap="Greys_r", vmin=vmin, vmax=vmax)
    

    date_str = f.stem.split("_")[2][:8] if len(f.stem.split("_")) > 2 else f.stem
    ax.text(0.05, 0.05, date_str, color="firebrick", fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])

# Turn off empty axes
for ax in axes[len(files):]:
    ax.axis("off")

# colorbar
cbar = fig_all.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
cbar.set_label("Backscatter (dB)", fontsize=11)

plt.suptitle(f"Sentinel-1 {polarization} backscatter over HB", fontsize=14, y=1.02)
plt.show()


# %%
n = len(files2)
cols = 7
rows = int(np.ceil(n / cols))
figsize_per_tile = 1.8
fig_all, axes = plt.subplots(rows, cols,
                             figsize=(cols * figsize_per_tile, rows * figsize_per_tile),
                             constrained_layout=True)
axes = axes.flatten()

#initialize value to keep the minimum and maximum values across all images.
#Since dB values include negative numbers, set +/- infinite.
vmin, vmax = np.inf, -np.inf 

for i, f in enumerate(files2): # give both the index (i) and the file path (f)
    with rasterio.open(f) as src: #open GeoTIFF using the rasterio library and automatically closes the file after reading
        sar2 = src.read(1).astype(float) # read 1st band and converts it to float (floats are needed for log10)
        sar2[sar2 <= 0] = np.nan # 'sar' contains backscatter values
        sar_dB2 = 10 * np.log10(sar2) # convert unit (desibel)
        extent = [ # defines the geographic information of the image (map coordinates).
            src.transform[2], #left (x_min)
            src.transform[2] + src.transform[0] * src.width, # move width * pixel size step to right (x_max) 
            src.transform[5] + src.transform[4] * src.height, # move height * pixel size step to bottom down (y_min)
            src.transform[5],# top (y_max)
        ]
    vmin = min(vmin, np.nanmin(sar_dB2))
    vmax = max(vmax, np.nanmax(sar_dB2))
    ax = axes[i]
    im = ax.imshow(sar_dB2, extent=extent, origin="upper", cmap="Greys_r", vmin=vmin, vmax=vmax)
    

    date_str = f.stem.split("_")[2][:8] if len(f.stem.split("_")) > 2 else f.stem
    ax.text(0.05, 0.05, date_str, color="firebrick", fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])

# Turn off empty axes
for ax in axes[len(files2):]:
    ax.axis("off")

# colorbar
cbar = fig_all.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
cbar.set_label("Backscatter (dB)", fontsize=11)

plt.suptitle(f"Sentinel-1 {polarization2} backscatter over HB", fontsize=14, y=1.02)
plt.show()


# %% Scatter plot 1: Snow depth vs backscatter
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

# === VV ===
ax = axes[0]
ax.scatter(df['snow_depth_cm'], df['VV'], color='black')
ax.set_title("(a) Snow depth vs VV", loc='left', fontsize=14)
ax.set_xlabel("Snow Depth (cm)")
ax.set_ylabel("VV backscatter (dB)")
ax.grid(True)

# === VH ===
ax = axes[1]
ax.scatter(df['snow_depth_cm'], df['VH'], color='black')
ax.set_title("(b) Snow depth vs VH", loc='left', fontsize=14)
ax.set_xlabel("Snow Depth (cm)")
ax.set_ylabel("VH backscatter (dB)")
ax.grid(True)

plt.tight_layout()
plt.show()


# %% Scatter plot 2: temperature threshold

# 1) Dry / Near-melt / Melt
dry      = df['TOBS.I-1 (degC) '] < -2
nearmelt = (df['TOBS.I-1 (degC) '] >= -2) & (df['TOBS.I-1 (degC) '] < 0)
melt     = df['TOBS.I-1 (degC) '] >= 0

# 2) Scatter plot: Depth vs Backscatter
fig = plt.figure(figsize=(12, 8))

# -----------------------------
# VV row (share y between VV)
# -----------------------------
ax_vv1 = plt.subplot2grid((2,3), (0,0))
ax_vv2 = plt.subplot2grid((2,3), (0,1), sharey=ax_vv1)
ax_vv3 = plt.subplot2grid((2,3), (0,2), sharey=ax_vv1)

# -----------------------------
# VH row (share y between VH)
# -----------------------------
ax_vh1 = plt.subplot2grid((2,3), (1,0))
ax_vh2 = plt.subplot2grid((2,3), (1,1), sharey=ax_vh1)
ax_vh3 = plt.subplot2grid((2,3), (1,2), sharey=ax_vh1)


ax = ax_vv1
ax.scatter(df.loc[dry,'snow_depth_cm'], df.loc[dry,'VV'], color='#1f77b4')

sub = df.loc[dry, ['snow_depth_cm','VV']].dropna() # polyfit purpose
ax.plot(sub['snow_depth_cm'], 
        np.poly1d(np.polyfit(sub['snow_depth_cm'], sub['VV'], 1))(sub['snow_depth_cm']),
        '--', lw=1, color='#1f77b4')

ax.set_title("(a) Dry (T < -2°C)", fontsize=14, loc='left')
ax.set_xlabel("Snow Depth (cm)")
ax.set_ylabel("VV backscatter (dB)")
ax.grid(True)


ax = ax_vv2
ax.scatter(df.loc[nearmelt,'snow_depth_cm'], df.loc[nearmelt,'VV'], color='#FF8C00')

sub = df.loc[nearmelt, ['snow_depth_cm','VV']].dropna() # polyfit purpose
ax.plot(sub['snow_depth_cm'], 
        np.poly1d(np.polyfit(sub['snow_depth_cm'], sub['VV'], 1))(sub['snow_depth_cm']),
        '--', lw=1, color='#FF8C00')

ax.set_title("(b) Near-melt (-2°C ≤ T < 0°C)", fontsize=14, loc='left')
ax.set_xlabel("Snow Depth (cm)")
ax.grid(True)


ax = ax_vv3
ax.scatter(df.loc[melt,'snow_depth_cm'], df.loc[melt,'VV'], color='crimson')

sub = df.loc[melt, ['snow_depth_cm','VV']].dropna() # polyfit purpose
ax.plot(sub['snow_depth_cm'], 
        np.poly1d(np.polyfit(sub['snow_depth_cm'], sub['VV'], 1))(sub['snow_depth_cm']),
        '--', lw=1, color='crimson')

ax.set_title("(c) Melt (T ≥ 0°C)", fontsize=14, loc='left')
ax.set_xlabel("Snow Depth (cm)")
ax.grid(True)


ax = ax_vh1
ax.scatter(df.loc[dry,'snow_depth_cm'], df.loc[dry,'VH'], color='#1f77b4')

sub = df.loc[dry, ['snow_depth_cm','VH']].dropna() # polyfit purpose
ax.plot(sub['snow_depth_cm'], 
        np.poly1d(np.polyfit(sub['snow_depth_cm'], sub['VH'], 1))(sub['snow_depth_cm']),
        '--', lw=1, color='#1f77b4')

ax.set_title("(d) Dry (T < -2°C)", fontsize=14, loc='left')
ax.set_xlabel("Snow Depth (cm)")
ax.set_ylabel("VH backscatter (dB)")
ax.grid(True)


ax = ax_vh2
ax.scatter(df.loc[nearmelt,'snow_depth_cm'], df.loc[nearmelt,'VH'], color='#FF8C00')

sub = df.loc[nearmelt, ['snow_depth_cm','VH']].dropna() # polyfit purpose
ax.plot(sub['snow_depth_cm'], 
        np.poly1d(np.polyfit(sub['snow_depth_cm'], sub['VH'], 1))(sub['snow_depth_cm']),
        '--', lw=1, color='#FF8C00')

ax.set_title("(e) Near-melt (-2°C ≤ T < 0°C)", fontsize=14, loc='left')
ax.set_xlabel("Snow Depth (cm)")
ax.grid(True)


ax = ax_vh3
ax.scatter(df.loc[melt,'snow_depth_cm'], df.loc[melt,'VH'], color='crimson')

sub = df.loc[melt, ['snow_depth_cm','VH']].dropna() # polyfit purpose
ax.plot(sub['snow_depth_cm'], 
        np.poly1d(np.polyfit(sub['snow_depth_cm'], sub['VH'], 1))(sub['snow_depth_cm']),
        '--', lw=1, color='crimson')

ax.set_title("(f) Melt (T ≥ 0°C)", fontsize=14, loc='left')
ax.set_xlabel("Snow Depth (cm)")
ax.grid(True)


plt.tight_layout()
plt.show()


# %%
df['VV_VH_diff'] = df['VV'] - df['VH']

fig = plt.figure(figsize=(5, 5))
ax = plt.gca()

ax.scatter(df['TOBS.I-1 (degC) '], df['VV_VH_diff'], color='black')

# polyfit for trend
sub = df[['TOBS.I-1 (degC) ', 'VV_VH_diff']].dropna()
coef = np.polyfit(sub['TOBS.I-1 (degC) '], sub['VV_VH_diff'], 1)
ax.plot(sub['TOBS.I-1 (degC) '], np.poly1d(coef)(sub['TOBS.I-1 (degC) ']),'r--', lw=1)

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("VV – VH (dB)")
ax.set_title("Polarization difference vs Temperature", loc='left', fontsize=14)
ax.grid(True)

plt.tight_layout()
plt.show()

# 온도 올라가면 눈에 물 생겨가지고 VV가 훨씬 크게 떨어진다
# VH는 상대적으로 덜 떨어진다 -> 좋은 관측!






















