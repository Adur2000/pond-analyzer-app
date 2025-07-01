import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Streamlit Page Config ---
st.set_page_config(page_title="🎣 Pond Analyzer Dashboard", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .css-18e3th9 { background-color: #ffffff; border-radius: 10px; padding: 2rem; }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #004c99;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #e9f5ff;
    }
    .stMetric {
        background-color: #e6f2ff;
        padding: 10px;
        border-radius: 10px;
        color: #003366;
        font-weight: bold;
    }
    .stDataFrame th {
        background-color: #eaf4ff !important;
    }
    .stDataFrame td {
        background-color: #f9fcff !important;
    }
    .stCaption {
        font-style: italic;
        color: #777;
    }
    </style>
""", unsafe_allow_html=True)

# --- User Login System ---
USER_CREDENTIALS = {
    "admin": {"password": "admin123", "role": "admin"},
    "viewer": {"password": "viewer123", "role": "viewer"},
    "Evance": {"password": "1234", "role": "admin"},
}

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["role"] = None

if not st.session_state["logged_in"]:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2023.png/600px-Logo_2023.png", width=150)
    st.markdown("# 🐟 Pond Analyzer Login")
    st.markdown("### Please enter your credentials to continue")
    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")
    login_btn = st.button("🔓 Login")

    if login_btn:
        username_input = username.strip().lower()
        credentials_lookup = {k.lower(): v for k, v in USER_CREDENTIALS.items()}

        if username_input in credentials_lookup and credentials_lookup[username_input]["password"] == password:
            st.session_state["logged_in"] = True
            st.session_state["role"] = credentials_lookup[username_input]["role"]
            st.success(f"✅ Logged in as {username} ({st.session_state['role']})")
            st.query_params["logged_in"] = "true"
        else:
            st.error("❌ Invalid credentials")
    st.stop()

# --- Main Dashboard Content ---
st.title("📊 Welcome to the Pond Performance Analyzer")
st.info("Upload your CSV data to begin analyzing pond performance metrics.")

uploaded_file = st.file_uploader("Upload Pond CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # --- Threshold Sliders ---
    st.sidebar.header("⚙️ Threshold Controls")
    fcr_threshold = st.sidebar.slider("FCR Alert Threshold", min_value=0.5, max_value=3.0, value=2.0, step=0.1)
    gpd_threshold = st.sidebar.slider("Growth per Day (GPD) Threshold", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    do_threshold = st.sidebar.slider("DO Alert Threshold (mg/L)", 0.0, 10.0, 4.0, 0.1)

    # --- KPI Calculations ---
    st.subheader("📌 Key Performance Indicators")
    if {'mortality_kg', 'mortality_count', 'stocked_number', 'feed_consumed_kg', 'total_weight_kg', 'days', 'do', 'saturation', 'nh3', 'no2'}.issubset(df.columns):
        df['survival_rate'] = ((df['stocked_number'] - df['mortality_count']) / df['stocked_number']) * 100
        df['mortality_rate'] = (df['mortality_count'] / df['stocked_number']) * 100
        df['avg_weight'] = (df['total_weight_kg'] / (df['stocked_number'] - df['mortality_count'])).round(2)
        df['fcr'] = (df['feed_consumed_kg'] / df['total_weight_kg']).round(2)
        df['gpd'] = (df['avg_weight'] / df['days']).round(2)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Avg. Weight (g)", f"{df['avg_weight'].mean():.2f}")
        col2.metric("Survival Rate (%)", f"{df['survival_rate'].mean():.2f}")
        col3.metric("FCR", f"{df['fcr'].mean():.2f}")
        col4.metric("GPD", f"{df['gpd'].mean():.2f}")
        col5.metric("DO (mg/L)", f"{df['do'].mean():.2f}")

        # Highlight anomalies
        df['fcr_flag'] = df['fcr'].apply(lambda x: 'High' if x > fcr_threshold else 'Normal')
        df['gpd_flag'] = df['gpd'].apply(lambda x: 'Low' if x < gpd_threshold else 'Normal')
        df['do_flag'] = df['do'].apply(lambda x: 'Low' if x < do_threshold else 'Normal')

    ...

    ...



    # --- Cluster Analysis ---
    st.subheader("🔍 Cluster Analysis")
    features = ['avg_weight', 'survival_rate', 'fcr', 'gpd', 'do', 'saturation', 'nh3', 'nitrate', 'nitrite']
    if all(col in df.columns for col in features):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[features])
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled_data)

        st.markdown("#### 🧬 Cluster Scatterplot")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='avg_weight', y='survival_rate', hue='cluster', palette='viridis', ax=ax)
        ax.set_title("Clusters based on Avg Weight vs Survival Rate")
        st.pyplot(fig)

        st.markdown("#### 📈 Line Chart (Cluster Trends)")
        fig_line, ax_line = plt.subplots()
        df.groupby('cluster')[features].mean().T.plot(kind='line', ax=ax_line)
        ax_line.set_title("Average KPI Trends by Cluster")
        st.pyplot(fig_line)

        st.markdown("#### 📊 Bar Chart (KPI Means by Cluster)")
        fig_bar, ax_bar = plt.subplots()
        df.groupby('cluster')[features].mean().plot(kind='bar', ax=ax_bar)
        ax_bar.set_title("Mean KPI Values by Cluster")
        ax_bar.set_ylabel("Value")
        st.pyplot(fig_bar)

        st.markdown("#### 🧩 Correlation Heatmap")
        corr = df[features].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        ax_corr.set_title("Correlation Matrix of KPIs")
        st.pyplot(fig_corr)

        st.markdown("#### 📉 Histogram of KPIs")
        for col in features:
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(df[col], bins=20, kde=True, ax=ax_hist)
            ax_hist.set_title(f"Distribution of {col}")
            st.pyplot(fig_hist)

    # --- Anomaly Detection ---
    st.subheader("🚨 Anomaly Detection")
    valid_features = [col for col in features if col in df.columns]
    if not valid_features:
        st.error("None of the selected features are present in the dataset.")
        st.stop()
    scaled_data = StandardScaler().fit_transform(df[valid_features])
    iso = IsolationForest(contamination=0.1)
    df['anomaly'] = iso.fit_predict(scaled_data)
    df['anomaly_label'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

    anomaly_counts = df['anomaly_label'].value_counts()
    st.write("Anomaly Summary:", anomaly_counts.to_dict())

    # Display Anomaly Table with Alert Flags
    st.markdown("#### 🧾 Anomaly Table with Alerts")
    def anomaly_flag(val):
        if val == 'Anomaly':
            return 'background-color: red; color: white'
        return ''

    anomaly_display = df[['pond_id', 'anomaly_label'] + valid_features].copy()
    styled_anomaly = anomaly_display.style.applymap(anomaly_flag, subset=['anomaly_label'])
    st.dataframe(styled_anomaly, use_container_width=True)

    # Optional download
    csv_data = anomaly_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Anomaly Table (CSV)",
        data=csv_data,
        file_name="anomaly_table.csv",
        mime="text/csv"
    )



    # --- Visualizations ---
    st.subheader("📊 Visualizations")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    col_x = st.selectbox("X-axis", numeric_cols, key='viz_x')
    col_y = st.selectbox("Y-axis", numeric_cols, index=1, key='viz_y')
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=col_x, y=col_y, hue='anomaly_label', style='cluster', palette='Set2', ax=ax)
    ax.set_title(f"{col_y} vs {col_x} (with Anomalies & Clusters)")
    st.pyplot(fig)

    # --- Interpretation ---
    st.subheader("📝 Interpretation")
    st.write("This visualization highlights pond health and performance influenced by environmental and management factors such as DO, ammonia, nitrite, FCR, and GPD.")



    # --- Report Download ---
    st.subheader("📥 Downloadable Report")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = "pond_reports"
    os.makedirs(folder_name, exist_ok=True)
    filename = f"pond_analysis_{timestamp}.csv"
    filepath = os.path.join(folder_name, filename)
    df.to_csv(filepath, index=False)
    st.success(f"✅ Report saved to {filepath}")
    st.download_button("Download CSV Report", df.to_csv(index=False).encode(), file_name=filename, mime="text/csv")

    # --- Correlation Matrix ---
    st.subheader("📈 Correlation Matrix")
    corr = df[features].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

 # --- Threshold Flags ---
st.subheader("⚠️ Threshold Flags")

# Define thresholds
fcr_thresh = st.sidebar.number_input("FCR Threshold", value=2.0, key="fcr_thresh")
gpd_thresh = st.sidebar.number_input("GPD Threshold", value=1.0, key="gpd_thresh")
do_thresh = st.sidebar.number_input("DO Threshold", value=4.0, key="do_thresh")

# Generate flag columns if not already created
if 'fcr' in df.columns:
    df['fcr_flag'] = df['fcr'].apply(lambda x: 'High' if x > fcr_thresh else 'OK' if not pd.isnull(x) else 'Missing')
if 'gpd' in df.columns:
    df['gpd_flag'] = df['gpd'].apply(lambda x: 'Low' if x < gpd_thresh else 'OK' if not pd.isnull(x) else 'Missing')
if 'do' in df.columns:
    df['do_flag'] = df['do'].apply(lambda x: 'Low' if x < do_thresh else 'OK' if not pd.isnull(x) else 'Missing')

flag_cols = ['fcr_flag', 'gpd_flag', 'do_flag']

# Highlight function
def highlight_flags(val):
    if val == 'High' or val == 'Low':
        return 'background-color: orange'
    elif val == 'Missing':
        return 'background-color: gray; color: white'
    return ''


    # --- Anomaly Detection ---
    st.subheader("🚨 Anomaly Detection")
    valid_features = [col for col in features if col in df.columns]
    if not valid_features:
        st.error("None of the selected features are present in the dataset.")
        st.stop()
    scaled_data = StandardScaler().fit_transform(df[valid_features])
    iso = IsolationForest(contamination=0.1)
    df['anomaly'] = iso.fit_predict(scaled_data)
    df['anomaly_label'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

    anomaly_counts = df['anomaly_label'].value_counts()
    st.write("Anomaly Summary:", anomaly_counts.to_dict())

    # Display Anomaly Table with Alert Flags
    st.markdown("#### 🧾 Anomaly Table with Alerts")
    def anomaly_flag(val):
        if val == 'Anomaly':
            return 'background-color: red; color: white'
        return ''

    anomaly_display = df[['pond_id', 'anomaly_label'] + valid_features].copy()
    styled_anomaly = anomaly_display.style.map(anomaly_flag, subset=['anomaly_label'])
    st.dataframe(styled_anomaly, use_container_width=True)

    # Optional download
    csv_data = anomaly_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Anomaly Table (CSV)",
        data=csv_data,
        file_name="anomaly_table.csv",
        mime="text/csv"
    )

    # --- Display Threshold Flags Table ---
    available_flags = [col for col in flag_cols if col in df.columns]
    display_cols = available_flags + ['cluster', 'anomaly_label'] if 'cluster' in df.columns else available_flags
    if display_cols:
        st.markdown("#### 📋 Threshold Flags Table")
        st.dataframe(df[display_cols].style.map(highlight_flags))

        # Save the flagged table to the report folder
        from datetime import datetime
        import os
        report_folder = "reports"
        os.makedirs(report_folder, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        flagged_path = os.path.join(report_folder, f"flagged_table_{date_str}.csv")
        df[display_cols].to_csv(flagged_path, index=False)

        # Show flag counts summary
        st.markdown("#### 🔢 Flag Summary Counts")
        for col in available_flags:
            counts = df[col].value_counts(dropna=False).to_dict()
            st.write(f"**{col}**:", counts)
    else:
        st.warning("None of the expected flag columns are available in the dataset.")
