
import pandas as pd
import folium
from folium.plugins import HeatMap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64, os

 
# Helper: figure → base64 PNG
 
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

 
# Region name lookup — assigns a state/region name from
# lat/lon using India's approximate state bounding boxes.
# Points that don't fall in any state are flagged as ocean.
 
INDIA_STATES = [
    # (name,          lat_min, lat_max, lon_min, lon_max)
    ("Jammu & Kashmir",  32.5,  37.0,  73.0,  80.0),
    ("Himachal Pradesh", 30.4,  33.2,  75.6,  79.0),
    ("Punjab",           29.5,  32.5,  73.9,  76.9),
    ("Uttarakhand",      28.7,  31.5,  77.6,  81.1),
    ("Haryana",          27.7,  30.9,  74.4,  77.6),
    ("Delhi",            28.4,  28.9,  76.8,  77.4),
    ("Rajasthan",        23.0,  30.2,  69.5,  78.3),
    ("Uttar Pradesh",    23.9,  30.4,  77.1,  84.6),
    ("Bihar",            24.3,  27.5,  83.3,  88.3),
    ("Sikkim",           27.0,  28.1,  88.0,  88.9),
    ("West Bengal",      21.5,  27.2,  85.8,  89.9),
    ("Assam",            24.1,  27.9,  89.7,  96.0),
    ("Meghalaya",        25.0,  26.1,  89.8,  92.8),
    ("Arunachal Pradesh",26.6,  29.5,  91.5,  97.4),
    ("Nagaland",         25.2,  27.0,  93.3,  95.2),
    ("Manipur",          23.8,  25.7,  93.0,  94.8),
    ("Mizoram",          21.9,  24.5,  92.3,  93.4),
    ("Tripura",          22.9,  24.5,  91.2,  92.3),
    ("Jharkhand",        21.9,  25.3,  83.3,  87.9),
    ("Odisha",           17.8,  22.6,  81.4,  87.5),
    ("Chhattisgarh",     17.8,  24.1,  80.2,  84.4),
    ("Madhya Pradesh",   21.1,  26.9,  74.0,  82.8),
    ("Gujarat",          20.1,  24.7,  68.2,  74.5),
    ("Maharashtra",      15.6,  22.0,  72.6,  80.9),
    ("Telangana",        15.8,  19.9,  77.2,  81.3),
    ("Andhra Pradesh",   12.6,  19.9,  76.8,  84.7),
    ("Karnataka",        11.6,  18.4,  74.1,  78.6),
    ("Goa",              14.9,  15.8,  73.9,  74.4),
    ("Kerala",            8.1,  12.8,  74.9,  77.4),
    ("Tamil Nadu",        8.1,  13.6,  76.2,  80.4),
]

def get_region_name(lat, lon):
    """Return state name if point is on land, else 'Ocean/Sea'."""
    for name, lat_min, lat_max, lon_min, lon_max in INDIA_STATES:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return None   # ocean / unrecognised

 
# 1. Load Data
 
df = pd.read_csv("data/processed/final_data.csv")
print(f" Loaded {len(df)} rows")

if "predicted_risk" not in df.columns:
    raise ValueError("❌ Run model.py first — 'predicted_risk' column missing!")

 
# 2. Tag each row with a region name, drop ocean points
 
df["region"] = df.apply(lambda r: get_region_name(r["latitude"], r["longitude"]), axis=1)

ocean_count = df["region"].isna().sum()
df = df[df["region"].notna()].copy()
print(f" Dropped {ocean_count} ocean/unrecognised points → {len(df)} land points remain")

COLOR_MAP = {0: "green", 1: "orange", 2: "red"}
LABEL_MAP = {0: "Low",   1: "Medium", 2: "High"}
BG_COLOR  = {0: "#e8f5e9", 1: "#fff8e1", 2: "#ffebee"}

 
# 3. Pre-build global distribution chart (shown in every popup)
 
risk_counts = df["predicted_risk"].value_counts().sort_index()
fig_g, ax_g = plt.subplots(figsize=(3, 2))
bars = ax_g.bar(
    [LABEL_MAP[i] for i in risk_counts.index],
    risk_counts.values,
    color=[COLOR_MAP[i] for i in risk_counts.index],
    edgecolor="white"
)
ax_g.set_title("Overall Risk Distribution", fontsize=8, fontweight="bold")
ax_g.set_ylabel("Count", fontsize=7)
ax_g.tick_params(labelsize=7)
ax_g.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, risk_counts.values):
    ax_g.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
              str(val), ha="center", va="bottom", fontsize=6)
global_chart_b64 = fig_to_base64(fig_g)

 
# 4. Create map
 
risk_map = folium.Map(
    location=[df["latitude"].mean(), df["longitude"].mean()],
    zoom_start=5,
    tiles="CartoDB positron"
)

# Heatmap layer
heat_data = df[["latitude", "longitude", "predicted_risk"]].values.tolist()
HeatMap(heat_data, radius=15, blur=20, name="Risk Heatmap").add_to(risk_map)

 
# 5. Add markers — sample for performance
 
sample_df = df.sample(n=min(300, len(df)), random_state=42)
total = len(sample_df)
features    = ["rainfall", "temperature", "humidity", "population_density"]
feat_labels = ["Rainfall", "Temp", "Humidity", "Pop.Density"]
max_vals    = [df[f].max() for f in features]
mean_vals   = [df[f].mean() for f in features]
mean_norm   = [v / m if m else 0 for v, m in zip(mean_vals, max_vals)]

for idx, (_, row) in enumerate(sample_df.iterrows()):
    risk_val = int(row["predicted_risk"])
    color    = COLOR_MAP[risk_val]
    label    = LABEL_MAP[risk_val]
    bg       = BG_COLOR[risk_val]
    region   = row["region"]

    if (idx + 1) % 50 == 0:
        print(f"   Building popups... {idx+1}/{total}")

    # Per-point comparison chart
    point_vals = [row[f] for f in features]
    point_norm = [v / m if m else 0 for v, m in zip(point_vals, max_vals)]

    fig_pt, ax2 = plt.subplots(figsize=(3.2, 2))
    x = range(len(feat_labels))
    ax2.bar([i - 0.2 for i in x], point_norm, width=0.35,
            color=color, alpha=0.85, label="This point")
    ax2.bar([i + 0.2 for i in x], mean_norm,  width=0.35,
            color="#90a4ae", alpha=0.85, label="Dataset avg")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(feat_labels, fontsize=6)
    ax2.set_title("Point vs Dataset Average", fontsize=8, fontweight="bold")
    ax2.set_ylabel("Normalised", fontsize=6)
    ax2.tick_params(labelsize=6)
    ax2.legend(fontsize=6)
    ax2.spines[["top", "right"]].set_visible(False)
    point_chart_b64 = fig_to_base64(fig_pt)

    popup_html = f"""
    <div style="
        font-family: Arial, sans-serif; font-size: 12px;
        width: 300px; background: {bg};
        border-radius: 8px; padding: 10px;
    ">
        <!-- Region name header -->
        <div style="font-size:16px; font-weight:bold; margin-bottom:2px;">
            📍 {region}
        </div>
        <div style="
            font-size: 14px; font-weight: bold; color: {color};
            margin-bottom: 6px; border-bottom: 2px solid {color}; padding-bottom: 4px;
        ">
            ⚠️ Risk Level: {label}
        </div>

        <table style="width:100%; border-collapse:collapse; margin-bottom:8px;">
            <tr>
                <td style="padding:2px 4px;">🌧 Rainfall</td>
                <td style="padding:2px 4px; font-weight:bold;">{row['rainfall']:.1f} mm</td>
                <td style="padding:2px 4px;">🌡 Temp</td>
                <td style="padding:2px 4px; font-weight:bold;">{row['temperature']:.1f} °C</td>
            </tr>
            <tr>
                <td style="padding:2px 4px;">💧 Humidity</td>
                <td style="padding:2px 4px; font-weight:bold;">{row['humidity']:.1f}%</td>
                <td style="padding:2px 4px;">👥 Pop.Density</td>
                <td style="padding:2px 4px; font-weight:bold;">{row['population_density']:.0f}</td>
            </tr>
            <tr>
                <td style="padding:2px 4px;">🏔 Elevation</td>
                <td style="padding:2px 4px; font-weight:bold;">{row['elevation']:.0f} m</td>
                <td style="padding:2px 4px;">🌊 River Disch.</td>
                <td style="padding:2px 4px; font-weight:bold;">{row['river_discharge']:.0f} m³/s</td>
            </tr>
        </table>

        <div style="font-size:10px; color:#555; margin-bottom:3px;">
            📊 <b>How this point compares to the dataset:</b>
        </div>
        <img src="data:image/png;base64,{point_chart_b64}"
             style="width:100%; border-radius:4px; margin-bottom:8px;"/>

        <div style="font-size:10px; color:#555; margin-bottom:3px;">
            🗺 <b>Overall risk distribution across all regions:</b>
        </div>
        <img src="data:image/png;base64,{global_chart_b64}"
             style="width:100%; border-radius:4px;"/>
    </div>
    """

    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=7,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.75,
        popup=folium.Popup(popup_html, max_width=320),
        tooltip=f"{region} — {label} Risk"   # shows state name on hover too
    ).add_to(risk_map)

print(f" {total} markers added")

 
# 6. Legend
 
legend_html = """
<div style="
    position: fixed; bottom: 40px; left: 40px;
    background: white; border: 2px solid #ccc;
    border-radius: 8px; padding: 10px 16px;
    font-family: Arial; font-size: 13px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.2); z-index: 9999;
">
    <b>Risk Level</b><br>
    <span style="color:red; font-size:18px;">&#9679;</span> High<br>
    <span style="color:orange; font-size:18px;">&#9679;</span> Medium<br>
    <span style="color:green; font-size:18px;">&#9679;</span> Low<br>
    <hr style="margin:6px 0;">
    <small>Click a point for details<br>Hover to see region name</small>
</div>
"""
risk_map.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl().add_to(risk_map)

 
# 7. Save
 
os.makedirs("outputs/maps", exist_ok=True)
risk_map.save("outputs/maps/risk_map.html")
print(" Map saved → outputs/maps/risk_map.html")