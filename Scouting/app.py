import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import os

# Load data
@st.cache_data
def load_data():
    filepath = os.path.join(os.path.dirname(__file__), "standardized_players.csv")
    return pd.read_csv(filepath)

df = load_data()

st.title("🔍 A-League Women Player Similarity Finder")

# ------------ SIDEBAR: SEARCH → SLIDER → FEATURE SELECTION ------------

# 1. 🔍 Search bar
selected_player = st.sidebar.selectbox("Search for a player:", df["Player"])

# 2. 🎚️ PCA dimension slider
# Assume stats start from column 5 onward; adjust if needed
all_stat_cols = df.columns[3:]
default_features = list(all_stat_cols)

# 3. 📊 Feature selection
selected_features = st.sidebar.multiselect(
    "Select stats to include in similarity search:",
    options=all_stat_cols,
    default=default_features
)

# 4. 🧠 Initialize PCA slider session state
if "pca_dims" not in st.session_state:
    st.session_state["pca_dims"] = 5

# Compute max allowable PCA dimensions
max_pca_dims = max(1, min(len(selected_features), len(df)))

# Clamp current value if it exceeds max
if st.session_state["pca_dims"] > max_pca_dims:
    st.session_state["pca_dims"] = max_pca_dims

# 5. 🎚️ Dynamic PCA slider
pca_slider_placeholder = st.sidebar.empty()
pca_dims = pca_slider_placeholder.slider(
    "Number of PCA dimensions",
    min_value=1,
    max_value=max_pca_dims,
    value=st.session_state["pca_dims"],
    key="pca_dims"
)


# ------------ MAIN APP LOGIC ------------

if selected_player and selected_features:
    features_df = df[selected_features]
    pca = PCA(n_components=pca_dims)
    features_reduced = pca.fit_transform(features_df.fillna(0))

    # Show variance captured
    explained = pca.explained_variance_ratio_.sum()
    st.sidebar.markdown(f"🧠 PCA captures **{explained:.1%}** of variance")

    # Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=6).fit(features_reduced)
    player_index = df[df["Player"] == selected_player].index[0]
    distances, indices = nbrs.kneighbors([features_reduced[player_index]])

    # Display similar players (excluding self)
    st.subheader(f"Players similar to **{selected_player}**")
    similar_players = df.iloc[indices[0][1:]][["Player", "Position_y", "Minutes played_y"]]
    similar_players = similar_players.rename(columns={
    "Player": "Player Name",
    "Position_y": "Position",
    "Minutes played_y": "Minutes Played"
    })

    st.dataframe(similar_players, use_container_width=True, hide_index=True)

else:
    st.info("Please select a player and at least one stat to begin.")
