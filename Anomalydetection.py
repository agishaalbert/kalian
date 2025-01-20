# ##############################################################################################################
# ########## ANOMALY DETECTION TIMESERIES KI- ALLIANZ - HS AALEN - HKM TEAM######################################
# #####################################################################################################


# To run the code in in your terminal .../ANOMALYDETECTIONHMTEAM$ streamlit run Anomalydetection.py 
# Some libraries needs to be installed to allow the model(TranAD) to run : check the requirements 

import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from plotly import graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import random
import time
from src.models import TranAD
#import os
#os.environ['DGLBACKEND'] = 'pytorch'



# Set random seed for reproducibility
def set_random_seed(seed=10000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    original_data = data.copy()
    
    if not pd.api.types.is_numeric_dtype(data.iloc[:, 0]):
        data = data.iloc[:, 1:]
    
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    return original_data, normalized_data, data.shape[1]

# Convert data to windows
def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    for i in range(len(data)):
        if i >= w_size:
            w = data[i-w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w)
    return torch.stack(windows)

# Detect anomalies using TranAD
def detect_anomalies_tranad(data, feats, percentile):
    set_random_seed()
    model = TranAD(feats).to(device)
    data_tensor = torch.tensor(data.values, dtype=torch.float).to(device)
    windows = convert_to_windows(data_tensor, model)
    
    model.eval()
    with torch.no_grad():
        losses, preds = [], []
        for window in windows:
            window = window.unsqueeze(0)
            elem = window[-1, :, :].view(1, window.shape[1], feats).to(device)
            z = model(window, elem)[1]
            loss = nn.MSELoss(reduction='none')(z, elem).mean(dim=[1, 2]).cpu().numpy()
            losses.append(loss)
            preds.append(z.squeeze(0).cpu().numpy())
    
    losses = np.array(losses).flatten()
    preds = np.array(preds)
    threshold = np.percentile(losses, percentile)
    anomalies = (losses > threshold).astype(int)
    return anomalies, losses, preds, threshold



def display_data(edited_data):
    data = edited_data
    data['Anomaly'] = data['Decision'].apply(lambda x: 1 if x == "Confirmed" else 0)

    confirmed_anomalies = data[data['Anomaly'] == 1]
    st.write(f"Confirmed Anomalies: {confirmed_anomalies['Anomaly'].sum()}", confirmed_anomalies)

    csv = data.to_csv(index=False)
    st.download_button("Download Updated Data", data=csv, file_name="updated_data.csv", mime="text/csv")


# Plot Up to 4 dimensions
def plot_4d_with_anomalies(data, anomalies, dimensions, time_column='Index'):
    """
    Plots up to 4 selected dimensions with anomalies.

    Parameters:
    - data: pd.DataFrame, normalized data containing the dimensions.
    - anomalies: np.array, binary array where 1 indicates an anomaly.
    - dimensions: list of str, the columns to plot.
    - time_column: str, index or time column for x-axis (default='Index').
    """
    if len(dimensions) > 4:
        st.warning("You can only plot up to 4 dimensions. The first 4 selected dimensions will be plotted.")
        dimensions = dimensions[:4]

    fig = go.Figure()
    colors = ['blue', 'green', 'orange', 'pink']

    for i, dim in enumerate(dimensions):
        fig.add_trace(go.Scatter(
            x=data.index if time_column == 'Index' else data[time_column],
            y=data[dim],
            mode='lines',
            name=dim,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
        

    # Add anomalies to all selected dimensions
    anomaly_indices = np.where(anomalies == 1)[0]
    for dim in dimensions:
        fig.add_trace(go.Scatter(
            x=anomaly_indices,
            y=data[dim].iloc[anomaly_indices],
            mode='markers',
            name=f'Anomalies on {dim}',
            marker=dict(color='red', size=8, symbol='star')
        ))

    # Layout customization
    fig.update_layout(
        title="Multi-Dimensional Time Series with Anomalies",
        xaxis_title=time_column,
        yaxis_title="Values",
        legend_title="Dimensions",
        template="plotly_dark",
        height=400,
        paper_bgcolor="#001f3f",  # Dark blue background
        plot_bgcolor="#001f3f",
    )

    return fig


# Update table with highlighted decisions 
@st.fragment
def update_table(data):
    
    with st.form('update_table'):
        cols = [x for x in data.columns if not x == 'Decision']
        # Style the decision column for display purposes
        styled_data = data.copy()
        styled_data['Decision'] = styled_data['Decision'].apply(
            lambda x: f"✅ Confirmed" if x == "Confirmed" else "❌ Rejected"
        )

        # Create an editable table
        edited_data = st.data_editor(
            styled_data,
            key="editable_data",
            use_container_width=True,
            column_config={
                "Decision": st.column_config.SelectboxColumn(
                    "Decision",
                    help="Select the decision for each point",
                    options=["✅ Confirmed", "❌ Rejected"],
                    required=True,
                )
            },
               disabled=cols,
        )

        # Clean up the decision column to remove styles for further processing
        edited_data['Decision'] = edited_data['Decision'].str.contains("✅").map(
            {True: "Confirmed", False: "Rejected"}
        )

        # Submit changes
        submit_btn = st.form_submit_button('Apply Changes')

    if submit_btn:
        display_data(edited_data)


# Main Function for the Anomaly Detection Streamlit App
def main():

    st.markdown("<style>body { text-align: center; }</style>", unsafe_allow_html=True)
    st.title("Anomaly Detection for Time Series - TranAD")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file:
            st.session_state.current_file = uploaded_file
            st.session_state.data = None
            st.session_state.anomalies = None
            st.session_state.scores = None
            st.session_state.preds = None

        original_data, normalized_data, feats = load_and_preprocess_data(uploaded_file)
        st.write("Original Data", original_data.head())

        threshold_percentile = st.slider("Set Anomaly Detection Threshold (Percentile)", 90, 99, 95)

        if st.button("Detect Anomalies"):
            with st.spinner('Detecting anomalies...'):
                start_time = time.time()
                anomalies, scores, preds, threshold = detect_anomalies_tranad(normalized_data, feats, threshold_percentile)
                end_time = time.time()

                st.session_state.data = original_data.copy()
                st.session_state.data['Anomaly'] = anomalies
                st.session_state.data['Decision'] = ['Confirmed' if a == 1 else 'Rejected' for a in anomalies]
                st.session_state.data['Anomaly_Score'] = scores
                st.session_state.anomalies = anomalies
                st.session_state.scores = scores
                st.session_state.preds = preds

                total_points = len(st.session_state.data)

                st.write(f"Detection completed in {end_time - start_time:.2f} seconds")
                st.write(f"Total Number of Points: {total_points}")
                st.write(f"Number of anomalies detected: {sum(anomalies)}")
                st.write(f"Threshold: {np.round(threshold, 4)}")

                st.write("Detected Anomalies (Editable)")
                update_table(st.session_state.data)

        if st.session_state.data is not None:
            st.markdown("### Multi-Dimensional Plot with Anomalies")
            available_dimensions = normalized_data.columns.tolist()
            selected_dimensions = st.multiselect(
                "Select up to 4 dimensions to plot", available_dimensions, default=available_dimensions[:4]
            )

            if selected_dimensions:
                fig = plot_4d_with_anomalies(
                    normalized_data,
                    st.session_state.anomalies,
                    selected_dimensions
                )
                st.plotly_chart(fig)

    st.markdown("---")
    st.markdown("© 2024 HM Team | HSAA")


if __name__ == "__main__":
    main()
