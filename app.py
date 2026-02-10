import os
import time
import warnings
from pathlib import Path

import streamlit as st
import folium
import leafmap.foliumap as leafmap
import rasterio
import numpy as np
from rasterio.warp import transform as rio_transform
from rasterio.crs import CRS
import plotly.graph_objects as go
from streamlit_folium import st_folium
import joblib
import pandas as pd

# Quiet GIS noise
warnings.filterwarnings("ignore")
os.environ.update({
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_LOG": "OFF",
    "PYTHONWARNINGS": "ignore",
})

st.set_page_config(page_title="Nairobi Flood Risk Dashboard", layout="wide")


# Session state

if "lat" not in st.session_state:
    st.session_state.lat = -1.2865
if "lon" not in st.session_state:
    st.session_state.lon = 36.8201
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "model" not in st.session_state:
    st.session_state.model = None


# Paths
APP_DIR = Path(__file__).resolve().parent          # .../webapp
PROJECT_ROOT = APP_DIR.parent

def resolve_raster_path(folder: str, file: str) -> Path:
    """
    Supports both layouts:
    1) ./layers/<folder>/<file>  (working directory)
    2) webapp/layers/<folder>/<file>
    3) project_root/layers/<folder>/<file>
    """
    p1 = Path("layers") / folder / file
    if p1.exists():
        return p1.resolve()

    p2 = APP_DIR / "layers" / folder / file
    if p2.exists():
        return p2

    p3 = PROJECT_ROOT / "layers" / folder / file
    if p3.exists():
        return p3

    return p2  # for error message

def boundary_geojson():
    p = APP_DIR / "assets" / "nairobi_boundary.geojson"
    return p if p.exists() else None


# Config
# Base risk thresholds - optimized based on actual data distribution
BASE_RISK_CLASSES = [
    ("Very Low",  0.0, 0.05, "#2E8B57"),   # Bottom ~30%
    ("Low",       0.05, 0.15, "#32CD32"),  # Next ~20%  
    ("Moderate",  0.15, 0.35, "#FFD700"),  # Middle range
    ("High",      0.35, 0.65, "#FF8C00"),  # Upper-middle
    ("Very High", 0.65, 1.0, "#DC143C"),   # Top ~15-20%
]

def get_risk_classes(sensitivity=1.0):
    """Get risk classes adjusted by sensitivity multiplier"""
    adjusted_classes = []
    for name, min_val, max_val, color in BASE_RISK_CLASSES:
        # Apply sensitivity - higher sensitivity = lower thresholds = more sensitive to risk
        adj_min = min_val / sensitivity if min_val > 0 else 0.0
        adj_max = max_val / sensitivity
        # Ensure we don't exceed 1.0
        adj_min = min(adj_min, 1.0)
        adj_max = min(adj_max, 1.0)
        adjusted_classes.append((name, adj_min, adj_max, color))
    return adjusted_classes

# Synthetic data enhanced based on notebook flood event patterns (0.01%-9.6% flood pixels)
LAYERS_CONFIG = {
    "Synthetic Scenarios": [
        {"label": "Low rainfall scenario (24h)", "file": "SYN_LOW_24H_enhanced.tif", "folder": "scenarios"},
        {"label": "Medium rainfall scenario (24h)", "file": "SYN_MED_24H_enhanced.tif", "folder": "scenarios"},
        {"label": "High rainfall scenario (24h)", "file": "SYN_HIGH_24H_enhanced.tif", "folder": "scenarios"},
        {"label": "Localised storm cell (24h)", "file": "SYN_STORMCELL_24H.tif", "folder": "scenarios"},
    ],
    "Historical Events": [
        {"label": "April 2016 flood event (best model)", "file": "E01_2016_04.tif", "folder": "events"},
        {"label": "May 2025 flood event (best model)", "file": "E15_2025_05.tif", "folder": "events"},
    ],
}

BASEMAPS = [
    "OpenStreetMap",
    "CartoDB.Positron",
    "CartoDB.DarkMatter",
    "Esri.WorldImagery",
]

PRESET_LOCATIONS = {
    "Custom": None,
    "Nairobi CBD": (-1.2865, 36.8201),
    "Westlands": (-1.2676, 36.8108),
    "Eastlands": (-1.2731, 36.8965),
    "Kibera": (-1.3133, 36.7886),
    "Industrial Area": (-1.3167, 36.8333),
    "Kasarani": (-1.2286, 36.8967),
    "Lang'ata": (-1.3642, 36.7520),
    "Embakasi": (-1.3106, 36.8919),
    "Karen": (-1.3197, 36.7086),
    "Runda": (-1.2088, 36.8086),
    "Gigiri": (-1.2288, 36.8019),
    "Parklands": (-1.2631, 36.8561),
    "South C": (-1.3256, 36.8378),
    "Kilimani": (-1.2906, 36.7822),
    "Upperhill": (-1.2947, 36.8108),
    "Huringham": (-1.3025, 36.7711),
    "Mathare": (-1.2531, 36.8614),
    "Kariobangi": (-1.2450, 36.8967),
    "Pipeline": (-1.3019, 36.8653),
    "Donholm": (-1.2833, 36.8978),
    "Buru Buru": (-1.2817, 36.8811),
    "Umoja": (-1.2700, 36.8933),
    "Komarock": (-1.2583, 36.9150),
    "Dandora": (-1.2467, 36.8892),
}

# Helpers
@st.cache_resource
def load_model():
    """Load the trained ML model"""
    try:
        model_path = Path(__file__).parent / "all_models.pkl"
        if model_path.exists():
            # Load the full model data structure
            models_data = joblib.load(model_path)
            
            # Extract the best model from nested structure
            if isinstance(models_data, dict) and 'models' in models_data and 'best_model_name' in models_data:
                actual_models = models_data['models']
                best_model_name = models_data['best_model_name']
                best_model = actual_models[best_model_name]
                
                # Also store feature names for later use
                feature_names = models_data.get('features', [])
                
                st.session_state.model = best_model
                st.session_state.model_features = feature_names
                
                st.success(f"✅ Loaded {best_model_name} model with {len(feature_names)} features")
                return best_model
            else:
                # Fallback for different model structure
                if isinstance(models_data, dict):
                    best_model = list(models_data.values())[0]
                else:
                    best_model = models_data
                st.session_state.model = best_model
                return best_model
        else:
            st.warning("Model file not found. Using fallback raster sampling.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def extract_features_at_point(lat: float, lon: float):
    """Extract features at a given point for model prediction"""
    try:
        # Get feature names from the loaded model
        feature_names = getattr(st.session_state, 'model_features', ['dem', 'slope', 'builtup', 'rain7', 'rain_sum'])
        
        features = {}
        
        # Map feature names to potential raster files
        feature_mapping = {
            'dem': ('static', 'elevation.tif'),
            'slope': ('static', 'slope.tif'),
            'builtup': ('static', 'builtup.tif'),
            'rain7': ('static', 'rain7.tif'),
            'rain_sum': ('static', 'rain_sum.tif'),
            'elevation': ('static', 'elevation.tif'),
            'drainage_density': ('static', 'drainage_density.tif'),
            'distance_to_water': ('static', 'distance_to_water.tif'),
            'soil_permeability': ('static', 'soil_permeability.tif'),
            'land_use': ('static', 'land_use.tif'),
        }
        
        # Extract features from rasters
        for feature_name in feature_names:
            if feature_name in feature_mapping:
                folder, filename = feature_mapping[feature_name]
                try:
                    raster_path = resolve_raster_path(folder, filename)
                    if raster_path and raster_path.exists():
                        with rasterio.open(raster_path) as src:
                            # Transform coordinates if needed
                            crs = src.crs
                            if crs and crs.to_epsg() != 4326:
                                x_arr, y_arr = rio_transform(CRS.from_epsg(4326), crs, [lon], [lat])
                                x, y = x_arr[0], y_arr[0]
                            else:
                                x, y = lon, lat
                            
                            val = next(src.sample([(x, y)]))[0]
                            if src.nodata is None or not np.isclose(val, src.nodata):
                                features[feature_name] = float(val)
                except Exception as e:
                    # Use default if extraction fails
                    pass
        
        # Add default values for missing features (based on model training data)
        default_features = {
            'dem': 1600.0,        # Typical Nairobi elevation  
            'slope': 5.0,         # Moderate slope
            'builtup': 0.3,       # Mixed urban/rural
            'rain7': 50.0,        # 7-day rainfall (mm)
            'rain_sum': 200.0,    # Total rainfall (mm)
            'elevation': 1600.0,
            'drainage_density': 0.5,
            'distance_to_water': 100.0,
            'soil_permeability': 0.3,
            'land_use': 2.0
        }
        
        # Fill in missing features with defaults
        for feature_name in feature_names:
            if feature_name not in features:
                features[feature_name] = default_features.get(feature_name, 0.0)
                
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return {}

def get_risk_band(value: float, sensitivity: float = 1.0):
    if value is None:
        return ("No Data", "#999999")
    risk_classes = get_risk_classes(sensitivity)
    for name, a, b, color in risk_classes:
        # include upper bound for last bin
        if name == "Very High" and value >= a:
            return (name, color)
        if a <= value < b:
            return (name, color)
    return ("No Data", "#999999")

@st.cache_data(show_spinner=False)
def raster_bounds_wgs84(raster_path_str: str):
    """Return (lat_min, lat_max, lon_min, lon_max) in WGS84 for validation."""
    rp = Path(raster_path_str)
    with rasterio.open(rp) as src:
        b = src.bounds
        crs = src.crs
        if crs is None:
            return (b.bottom, b.top, b.left, b.right)

        epsg = crs.to_epsg() if crs else None
        if epsg == 4326:
            return (b.bottom, b.top, b.left, b.right)

        xs = [b.left, b.right, b.left, b.right]
        ys = [b.bottom, b.bottom, b.top, b.top]
        lon, lat = rio_transform(crs, CRS.from_epsg(4326), xs, ys)
        return (min(lat), max(lat), min(lon), max(lon))

def in_bounds(lat, lon, bounds):
    lat_min, lat_max, lon_min, lon_max = bounds
    return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)

@st.cache_data(show_spinner=False)
def sample_raster_value(raster_path_str: str, lat: float, lon: float):
    """
    Sample using ML model prediction with rainfall scenario adjustments.
    """
    # Load model if not already loaded
    if st.session_state.model is None:
        st.session_state.model = load_model()
    
    model = st.session_state.model
    
    if model is not None:
        try:
            # Extract environmental features at the point
            features = extract_features_at_point(lat, lon)
            
            if features:
                # Get the expected feature names from the model
                feature_names = getattr(st.session_state, 'model_features', list(features.keys()))
                
                # Create feature array in the correct order
                feature_values = [features.get(name, 0.0) for name in feature_names]
                feature_df = pd.DataFrame([feature_values], columns=feature_names)
                
                # Make prediction (returns probability)
                pred_proba = model.predict_proba(feature_df)
                flood_prob = pred_proba[0][1] if len(pred_proba[0]) > 1 else pred_proba[0][0]
                
                rp = Path(raster_path_str)
                filename = rp.name.upper()
                
                if 'LOW' in filename:
                    
                    flood_prob *= 0.7
                elif 'MED' in filename:
                    
                    flood_prob *= 1.0  
                elif 'HIGH' in filename:
                    
                    flood_prob *= 1.3
                
                
                enhanced_prob = min(flood_prob * 3.0, 1.0)
                
                return enhanced_prob
                
        except Exception as e:
            st.warning(f"Model prediction failed: {e}. Using fallback method.")
    
    # Fallback to raster sampling if model fails or unavailable
    rp = Path(raster_path_str)
    try:
        with rasterio.open(rp) as src:
            crs = src.crs
            if crs is None:
                x, y = lon, lat
            else:
                epsg = crs.to_epsg() if crs else None
                if epsg == 4326:
                    x, y = lon, lat
                else:
                    x_arr, y_arr = rio_transform(CRS.from_epsg(4326), crs, [lon], [lat])
                    x, y = x_arr[0], y_arr[0]

            val = next(src.sample([(x, y)]))[0]
            if src.nodata is not None and np.isclose(val, src.nodata):
                return None
            if np.isnan(val):
                return None
                
            # Probability scaling based on scenario type
            enhanced_val = min(float(val) * 3.0, 1.0)
            
            # Ensure logical rainfall-flood relationship
            filename = rp.name.upper()
            if 'LOW' in filename:
                enhanced_val *= 0.7
            elif 'MED' in filename:
                enhanced_val *= 1.0  
            elif 'HIGH' in filename:
                enhanced_val *= 1.3
                
            return min(enhanced_val, 1.0)
    except:
        return None

def scenario_values_at_point(lat: float, lon: float):
    """Values across all synthetic scenarios at this point."""
    out = []
    for item in LAYERS_CONFIG["Synthetic Scenarios"]:
        p = resolve_raster_path(item["folder"], item["file"])
        if p.exists():
            v = sample_raster_value(str(p), lat, lon)
            out.append((item["label"], v))
        else:
            out.append((item["label"], None))
    return out

def risk_band_strip_figure(value: float, sensitivity: float = 1.0):
    """
    Horizontal strip of the 5 risk bands with a marker at the selected probability.
    Risk bands adjust based on sensitivity setting.
    """
    # Build one stacked horizontal bar with segments
    fig = go.Figure()
    risk_classes = get_risk_classes(sensitivity)

    # Start from 0.0; each segment has variable length
    for name, a, b, color in risk_classes:
        fig.add_trace(
            go.Bar(
                x=[b - a],
                y=["Risk bands"],
                orientation="h",
                marker=dict(color=color),
                name=f"{name} ({a:.1f}–{b:.1f})",
                hovertemplate=f"{name}: {a:.1f}–{b:.1f}<extra></extra>"
            )
        )

    # Add marker line for the selected probability
    if value is not None:
        fig.add_vline(
            x=value,
            line_width=3,
            line_dash="dash",
            line_color="black",
            annotation_text=f"{value:.3f}",
            annotation_position="top"
        )

    fig.update_layout(
        barmode="stack",
        height=220,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(range=[0, 1], title="Probability / susceptibility"),
        yaxis=dict(title="", showticklabels=False),
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
        title="Probability compared against risk bands"
    )
    return fig


# UI
st.markdown(
    "<h1 style='text-align:center;margin-bottom:0.25rem;'>Nairobi Flood Risk Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;margin-top:0;color:#888;'>Flood risk mapping to support planning and scenario testing across Nairobi County.</p>",
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("Controls")

    # Default to Synthetic Scenarios
    groups = list(LAYERS_CONFIG.keys())
    if "Synthetic Scenarios" in groups:
        groups.remove("Synthetic Scenarios")
        groups.insert(0, "Synthetic Scenarios")

    layer_group = st.selectbox("Layer group", groups, index=0)

    layers = LAYERS_CONFIG[layer_group]
    layer_labels = [x["label"] for x in layers]
    selected_label = st.selectbox("Select layer", layer_labels, index=0)
    selected_layer = next(x for x in layers if x["label"] == selected_label)

    basemap = st.selectbox("Basemap", BASEMAPS, index=0)
    opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.70, 0.05)

    # Sensitivity Control
    st.divider()
    st.subheader("Risk Sensitivity")
    sensitivity = st.slider(
        "Sensitivity multiplier", 
        min_value=0.5, 
        max_value=2.0, 
        value=1.0, 
        step=0.1,
        help="Higher sensitivity = more areas classified as risky. Lower sensitivity = more conservative assessment."
    )
    
    if sensitivity != 1.0:
        if sensitivity > 1.0:
            st.info(f"🔍 **High sensitivity** ({sensitivity:.1f}x): More areas show as risky")
        else:
            st.info(f"🛡️ **Conservative** ({sensitivity:.1f}x): Only highest risk areas highlighted")

    show_boundary = st.checkbox("Show Nairobi boundary", value=True)

    st.divider()
    st.subheader("Location")

    preset = st.selectbox("Quick locations", list(PRESET_LOCATIONS.keys()), index=0)
    if preset != "Custom":
        st.session_state.lat, st.session_state.lon = map(float, PRESET_LOCATIONS[preset])

    lat = st.number_input("Latitude", value=float(st.session_state.lat), format="%.6f")
    lon = st.number_input("Longitude", value=float(st.session_state.lon), format="%.6f")

    st.session_state.lat = float(lat)
    st.session_state.lon = float(lon)

    analyse = st.button("Analyse", type="primary")

# Resolve raster
rp = resolve_raster_path(selected_layer["folder"], selected_layer["file"])
raster_exists = rp.exists()

# Validate bounds
bounds = None
valid_point = False
if raster_exists:
    bounds = raster_bounds_wgs84(str(rp))
    valid_point = in_bounds(lat, lon, bounds)

# Map
st.subheader("Interactive Map")

# Use leafmap directly with to_streamlit method
m = leafmap.Map(center=[lat, lon], zoom=11, height="480px")

# Add basemap
try:
    m.add_basemap(basemap)
except Exception:
    pass  # Use default

# Add marker
m.add_marker([lat, lon], popup=f"Analysis Point<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}")

# Add boundary if enabled
if show_boundary:
    # Add rectangle boundary
    bounds_coords = [
        [[-1.45, 36.60], [-1.45, 37.10], [-1.15, 37.10], [-1.15, 36.60], [-1.45, 36.60]]
    ]
    try:
        m.add_geojson({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": bounds_coords
                },
                "properties": {"name": "Data Coverage Area"}
            }]
        }, style={"color": "red", "weight": 2, "fillOpacity": 0})
    except Exception:
        pass

# Add raster overlay if exists
if raster_exists:
    try:
        m.add_raster(str(rp), opacity=float(opacity), layer_name=selected_layer["label"])
    except Exception as e:
        st.warning(f"Could not load raster overlay: {str(e)[:100]}")

# Display map
m.to_streamlit(height=480)

# Status row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.write(f"**Layer:** {selected_layer['label']}")
with c2:
    st.write(f"**Group:** {layer_group}")
with c3:
    st.write(f"**Point:** {lat:.4f}, {lon:.4f}")
with c4:
    st.write("**Status:** Ready" if st.session_state.analysis_results is None else "**Status:** Analysed")

# Coverage message
if raster_exists and bounds is not None:
    if valid_point:
        st.success("Point is inside the selected layer coverage.")
    else:
        lat_min, lat_max, lon_min, lon_max = bounds
        st.warning("Point is outside the selected layer coverage.")
        st.info(f"Valid area: {lat_min:.4f}–{lat_max:.4f} lat, {lon_min:.4f}–{lon_max:.4f} lon")

# Analyse
if analyse:
    if not raster_exists:
        st.error("Cannot analyse: selected raster file is missing.")
        st.session_state.analysis_results = None
    elif not valid_point:
        st.error("Cannot analyse: point is outside this layer coverage.")
        st.session_state.analysis_results = None
    else:
        with st.spinner("Analysing point..."):
            time.sleep(0.15)
            v = sample_raster_value(str(rp), float(lat), float(lon))

        if v is None:
            st.warning("No data at this point (nodata). Try a nearby point.")
            st.session_state.analysis_results = None
        else:
            band_name, band_color = get_risk_band(v, sensitivity)
            st.session_state.analysis_results = {
                "risk_value": float(v),
                "risk_band": band_name,
                "risk_color": band_color,
                "coords": (float(lat), float(lon)),
                "layer": selected_layer["label"],
            }

# Results
if st.session_state.analysis_results:
    st.divider()
    st.subheader("Point result")

    res = st.session_state.analysis_results
    risk_value = res["risk_value"]
    risk_band = res["risk_band"]
    risk_color = res["risk_color"]

    r1, r2 = st.columns(2)
    with r1:
        st.metric("Flood probability / susceptibility", f"{risk_value:.3f}")
    with r2:
        st.markdown(
            f"**Risk band:** <span style='color:{risk_color};font-weight:800;font-size:1.1rem'>{risk_band}</span>",
            unsafe_allow_html=True
        )

    # Gauge (kept)
    fig_g = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_value,
            title={"text": "Flood risk gauge"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": risk_color},
                "steps": [{"range": [a, b], "color": c} for (_, a, b, c) in get_risk_classes(sensitivity)],
            },
        )
    )
    fig_g.update_layout(height=300, margin=dict(l=20, r=20, t=45, b=10))
    st.plotly_chart(fig_g, width="stretch")

    # REQUIRED: Risk-band comparison strip plot
    st.plotly_chart(risk_band_strip_figure(risk_value, sensitivity), width="stretch")

    # Optional: scenario comparison at this point (stakeholder-friendly)
    st.markdown("### Scenario comparison at this point")
    scen = scenario_values_at_point(res["coords"][0], res["coords"][1])
    scen_labels = [s[0] for s in scen]
    scen_vals = [(s[1] if s[1] is not None else np.nan) for s in scen]

    fig_s = go.Figure(data=[go.Bar(x=scen_labels, y=scen_vals)])
    fig_s.update_layout(
        yaxis=dict(range=[0, 1], title="Probability / susceptibility"),
        xaxis=dict(title="Synthetic scenario"),
        height=320,
        margin=dict(l=20, r=20, t=30, b=30),
        title="How the probability changes across scenarios"
    )
    st.plotly_chart(fig_s, width="stretch")
    
    # Stakeholder Recommendations - "So What?"
    st.markdown("---")
    st.markdown("### 📋 Strategic Recommendations for Stakeholders")
    
    # Analyze scenario results for recommendations
    valid_scenarios = [s for s in scen if s[1] is not None]
    if valid_scenarios:
        scenario_risks = [s[1] for s in valid_scenarios]
        avg_risk = np.mean(scenario_risks)
        max_risk = max(scenario_risks)
        min_risk = min(scenario_risks)
        risk_range = max_risk - min_risk
        
        # Risk level assessment
        if avg_risk < 0.2:
            risk_level = "LOW"
            color = "#2E8B57"
        elif avg_risk < 0.4:
            risk_level = "MODERATE-LOW"
            color = "#32CD32"
        elif avg_risk < 0.6:
            risk_level = "MODERATE"
            color = "#FFD700"
        elif avg_risk < 0.8:
            risk_level = "HIGH"
            color = "#FF8C00"
        else:
            risk_level = "VERY HIGH"
            color = "#DC143C"
        
        # Main recommendation panel
        st.markdown(f"""
        <div style='padding: 1.5rem; background-color: {color}20; border-left: 4px solid {color}; border-radius: 5px; margin: 1rem 0;'>
            <h4 style='color: {color}; margin: 0;'>📊 Risk Assessment Summary</h4>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
                <strong>Overall Risk Level:</strong> {risk_level}<br>
                <strong>Average Scenario Risk:</strong> {avg_risk:.3f}<br>
                <strong>Risk Variability:</strong> {risk_range:.3f} (Range: {min_risk:.3f} - {max_risk:.3f})
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Specific recommendations based on risk level
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown("**🏗️ Development & Planning:**")
            if avg_risk < 0.3:
                st.write("📝 Suitable for most development types")
                st.write("📝 Standard drainage requirements")
                st.write("📝 Normal building codes sufficient")
            elif avg_risk < 0.6:
                st.write("⚠️ Enhanced drainage systems recommended")
                st.write("⚠️ Flood-resistant construction methods")
                st.write("⚠️ Consider elevation requirements")
            else:
                st.write("🚫 High-risk area - avoid critical infrastructure")
                st.write("🚫 Extensive flood protection required")
                st.write("🚫 Consider alternative locations")
        
        with rec_col2:
            st.markdown("**💼 Investment & Insurance:**")
            if avg_risk < 0.3:
                st.write("💰 Standard insurance rates expected")
                st.write("💰 Good investment potential")
                st.write("💰 Minimal flood risk premiums")
            elif avg_risk < 0.6:
                st.write("💰 Flood insurance highly recommended")
                st.write("💰 Factor risk into property valuations")
                st.write("💰 Budget for mitigation measures")
            else:
                st.write("💰 High insurance premiums likely")
                st.write("💰 Significant investment in protection needed")
                st.write("💰 Consider risk vs. return carefully")
        
        # Scenario-specific insights
        st.markdown("**🌧️ Climate Scenario Insights:**")
        
        high_risk_scenarios = [s for s in valid_scenarios if s[1] > 0.6]
        if high_risk_scenarios:
            st.warning(f"**High Risk Alert:** {len(high_risk_scenarios)} scenario(s) show significant flood risk:")
            for scenario_name, risk_val in high_risk_scenarios:
                st.write(f"   • {scenario_name}: {risk_val:.3f} probability")
        
        if risk_range > 0.3:
            st.info("**Variable Risk:** Flood susceptibility varies significantly across scenarios. Consider adaptive management strategies.")
        
        # Action items
        st.markdown("**📋 Recommended Next Steps:**")
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("📊 Generate Full Report", help="Create comprehensive risk assessment report"):
                st.success("Detailed risk report generation coming soon!")
        
        with action_col2:
            if st.button("👥 Consult Experts", help="Connect with flood risk professionals"):
                st.success("Expert consultation booking coming soon!")
        
        with action_col3:
            if st.button("🏛️ Policy Guidance", help="Get regulatory and policy recommendations"):
                st.success("Policy guidance portal coming soon!")

# Download selected raster
if raster_exists:
    try:
        with open(rp, "rb") as f:
            st.download_button(
                "Download selected raster (GeoTIFF)",
                data=f.read(),
                file_name=rp.name,
                mime="image/tiff",
                width="stretch",
            )
    except Exception:
        pass
