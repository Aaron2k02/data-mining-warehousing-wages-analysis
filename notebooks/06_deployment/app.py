import streamlit as st
import h2o
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gdown

# Initialize H2O
h2o.init()

# Download files from Google Drive
def download_file_from_google_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# Download the model file
model_file_id = "1EEZ490OTSDsnovwb2NJTW1_XlnRyD35X"
model_path = "DeepLearning_model_python_1735461052018_31"

# Debugging: Check if the model file exists
if os.path.exists(model_path):
    st.write(f"Model file found at: {model_path}")
else:
    st.write(f"Model file not found at: {model_path}. Downloading...")
    try:
        download_file_from_google_drive(model_file_id, model_path)
        st.write("Model file downloaded successfully.")
    except Exception as e:
        st.error(f"Error downloading model file: {e}")

# Load the H2O Deep Learning model
@st.cache_resource
def load_model():
    try:
        model = h2o.load_model(model_path)
        st.write("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Streamlit app
st.title("H2O Deep Learning Model Deployment")
st.write("Upload your input data (CSV) and get predictions!")

# Add a sidebar for guidance
st.sidebar.title("User Guide")
st.sidebar.write("""
### How to Use This App
1. **Upload a CSV File**:
   - Click the "Browse files" button or drag and drop a CSV file into the uploader.
   - The CSV file should contain the same features (columns) as the training data used for the model.

2. **Make Predictions**:
   - After uploading the file, click the "Predict" button.
   - The app will generate predictions and display them in a table.

3. **Compare Predictions with Actual Values (Optional)**:
   - If you upload `X_val.csv` or `X_test.csv`, the app will automatically load the corresponding `y_val.csv` or `y_test.csv` to compare predictions with actual values.
   - Evaluation metrics (RMSE, MAE, R²) and visualizations (Actual vs Predicted Values, Residuals Plot) will be displayed.

4. **Download Predictions**:
   - You can download the predictions as a CSV file by clicking the "Download Predictions" button.

### Notes
- Ensure the uploaded CSV file has the same format as the training data.
- If the app displays an error, check the file format and try again.
""")

# Upload input file
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded CSV file
    input_df = pd.read_csv(uploaded_file)

    # Convert the input data to H2OFrame
    input_h2o = h2o.H2OFrame(input_df)

    # Make predictions
    if model is not None:
        try:
            predictions = model.predict(input_h2o)
            predictions_df = predictions.as_data_frame()

            # Display predictions
            st.write("Predictions:")
            st.write(predictions_df)

            # Add a download button for predictions
            st.download_button(
                label="Download Predictions",
                data=predictions_df.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )

            # Add visualizations for predictions
            st.write("### Visualizations for Predictions")

            # Histogram of predictions (Plotly)
            st.write("#### Distribution of Predictions:")
            fig = px.histogram(
                predictions_df, 
                x="predict", 
                nbins=20, 
                title="Histogram of Predictions",
                color_discrete_sequence=["skyblue"]
            )
            st.plotly_chart(fig)

            # Line plot of predictions (Plotly)
            st.write("#### Line Plot of Predictions:")
            fig = px.line(
                predictions_df, 
                y="predict", 
                title="Line Plot of Predictions",
                color_discrete_sequence=["green"]
            )
            st.plotly_chart(fig)

            # Top N predictions
            top_n = st.slider("#### Select the number of top predictions to display", min_value=5, max_value=50, value=10)
            st.write(f"##### Top {top_n} Predictions:")
            top_predictions = predictions_df.nlargest(top_n, "predict")
            st.write(top_predictions)

            # Bar chart of top N predictions (Plotly)
            st.write(f"#### Bar Chart of Top {top_n} Predictions:")
            fig = px.bar(
                top_predictions, 
                x=top_predictions.index, 
                y="predict", 
                title=f"Top {top_n} Predictions",
                color_discrete_sequence=["purple"]
            )
            st.plotly_chart(fig)

            # Load actual target values
            if "X_val.csv" in uploaded_file.name:
                y_val_file_id = "1rMg9X3mpKAhYfPGBa06qq0CUTRvQaJ_2"
                y_val_path = "y_val.csv"
                if not os.path.exists(y_val_path):
                    download_file_from_google_drive(y_val_file_id, y_val_path)
                actual_values = pd.read_csv(y_val_path)
            elif "X_test.csv" in uploaded_file.name:
                y_test_file_id = "1bBEybz9U3rjOcFtdQa-Tt54xRqZLVc94"
                y_test_path = "y_test.csv"
                if not os.path.exists(y_test_path):
                    download_file_from_google_drive(y_test_file_id, y_test_path)
                actual_values = pd.read_csv(y_test_path)
            else:
                actual_values = None

            # Compare predictions with actual values
            if actual_values is not None:
                # Ensure actual_values is a 1D array or Series
                if actual_values.ndim > 1:
                    actual_values = actual_values.squeeze()

                # Check if predictions and actual values have the same length
                if len(predictions_df) != len(actual_values):
                    st.error(
                        f"Mismatch in lengths: Predictions ({len(predictions_df)}) vs Actual Values ({len(actual_values)})")
                else:
                    # Calculate evaluation metrics
                    mse = mean_squared_error(actual_values, predictions_df["predict"])
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(actual_values, predictions_df["predict"])
                    r2 = r2_score(actual_values, predictions_df["predict"])

                    # Display evaluation metrics
                    st.write("#### Evaluation Metrics:")
                    st.write(f"- RMSE: {rmse:.4f}")
                    st.write(f"- MAE: {mae:.4f}")
                    st.write(f"- R²: {r2:.4f}")

                    # Visualize predictions vs actual values (Plotly)
                    st.write("#### Predictions vs Actual Values:")
                    fig = px.scatter(
                        x=actual_values, 
                        y=predictions_df["predict"], 
                        labels={"x": "Actual Values", "y": "Predicted Values"},
                        title="Actual vs Predicted Values",
                        color_discrete_sequence=["blue"]
                    )
                    st.plotly_chart(fig)

                    # Visualize residuals (Plotly)
                    st.write("#### Residuals Plot:")
                    residuals = actual_values - predictions_df["predict"]
                    fig = px.scatter(
                        x=predictions_df["predict"], 
                        y=residuals, 
                        labels={"x": "Predicted Values", "y": "Residuals"},
                        title="Residuals vs Predicted Values",
                        color_discrete_sequence=["blue"]
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig)

                    # Compare histograms of predictions and actual values (Plotly)
                    st.write("#### Histogram of Predictions vs Actual Values:")
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=predictions_df["predict"], 
                        name="Predictions", 
                        marker_color="skyblue",
                        opacity=0.75
                    ))
                    fig.add_trace(go.Histogram(
                        x=actual_values, 
                        name="Actual Values", 
                        marker_color="orange",
                        opacity=0.75
                    ))
                    fig.update_layout(
                        barmode="overlay", 
                        title="Histogram of Predictions vs Actual Values",
                        xaxis_title="Values",
                        yaxis_title="Frequency"
                    )
                    st.plotly_chart(fig)

                    # Interpretation for histograms
                    st.write("""
                    **Interpretation of Histograms**:
                    - The **blue bars** represent the distribution of **predicted values**.
                    - The **orange bars** represent the distribution of **actual values**.
                    - If the distributions overlap well, the model is capturing the data accurately.
                    - If the distributions differ significantly, the model might need improvement.
                    """)
        except Exception as e:
            st.error(f"Error making predictions: {e}")
    else:
        st.error("Model is not loaded. Please check the model file.")