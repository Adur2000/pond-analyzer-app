# pond_analyzer_app.py

import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import random
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from PIL import Image
import io
import openai



# --- OpenAI Client Setup ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_ai_insight(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You're an aquaculture data analyst. Explain charts and detect patterns."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

# --- Styling ---
st.set_page_config(page_title="🎣 Pond Analyzer Dashboard", layout="wide")

# --- Login ---
USER_CREDENTIALS = {
    "admin": {"password": "admin123"},
    "Evance": {"password": "1234"}
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🐟 Pond Analyzer Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username]["password"] == password:
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# Streamlit-based Pond Performance Investor Dashboard with Multi-Tab Design
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os
from io import BytesIO

from datetime import datetime

st.set_page_config(layout="wide")

st.title("📈 Aquaculture  Dashboard")
with st.sidebar:
    st.title("📊 Insights Dashboard")
    st.markdown("""
    **Tabs Guide**  
    - 📈 Trends: View growth and feed conversion  
    - 🧮 Visualizations: See how KPIs relate  
    - ⚠️ Health: Risk flags from thresholds  
    - 📋 Summary: Daily metrics overview  
    - 🧬 Disease Detection: Image & symptom-based analysis
    """)
    st.markdown("---")
    st.info("📞 Contact us: aquadata@evanceotieno959@gmail.com",)


# Upload CSV
# Upload and load the dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Ensure 'pond_id' and 'date' exist
    if 'pond_id' in df.columns and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])  # Convert to datetime

        # Now you can safely filter
        pond_filter = st.multiselect("Select pond(s) to visualize", options=df['pond_id'].unique(), default=df['pond_id'].unique(), key="pond_filter_trend")
        if not pond_filter:
            st.warning("Please select at least one pond to visualize.")
        else:
            # Filter the DataFrame based on selected ponds
            st.write(f"Showing data for ponds: {', '.join(pond_filter)}")

        filtered_df = df[df['pond_id'].isin(pond_filter)]

        # Example chart
        st.line_chart(filtered_df.set_index("date")[['fcr', 'gpd']])

    else:
        st.error("Missing 'pond_id' or 'date' column in the uploaded dataset.")
else:
    st.warning("Please upload a CSV file.")


    
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Trend", "Visualizations", "Health", "Summary", " Disease Detection"])

with tab1:
    st.subheader("\U0001F4CA Daily Trends")
    metric = st.selectbox("Choose metric to visualize", ['gpd', 'fcr', 'do', 'survival_rate', 'avg_weight'])
    pond_filter = st.multiselect("Select pond(s) to visualize", options=df['pond_id'].unique(), default=df['pond_id'].unique())
    smoothing = st.checkbox("Apply 7-day smoothing", value=True)
    from datetime import date

with st.expander("📝 Enter Feed Data Manually"):
    FEED_TYPES = sorted(list(set([
        "LFL Mauritius Starter big crumble",
        "LFL Mauritius Starter small crumble",
        "Tunga Egypt Nutra 120",
        "Tunga Egypt Nutra 0",
        "Tunga Egypt  Nutra 80",
        "Tunga Egypt Nutra 160",
        "Samakgro Kenya Tilapia 2mm",
        "Samakgro Kenya Tilapia 3mm",
        "Samakgro Kenya Tilapia SP4LV724 4mm"
    ])))

    if "manual_feed_data" not in st.session_state:
        st.session_state.manual_feed_data = []

    with st.form("manual_feed_entry_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            pond_id = st.text_input("Pond ID")
            feed_quantity = st.number_input("Feed Given (kg)", min_value=0.0, step=0.1)
        with col2:
            entry_date = st.date_input("Date", value=date.today())
            feed_type = st.selectbox("Feed Type", FEED_TYPES)

        submit = st.form_submit_button("➕ Add Entry")

        if submit:
            st.session_state.manual_feed_data.append({
                "pond_id": pond_id,
                "date": entry_date,
                "feed_type": feed_type,
                "feed_quantity": feed_quantity
            })
            st.success("✅ Feed entry added successfully!")

    if st.session_state.manual_feed_data:
        st.markdown("### 📋 New Feed Entries")
        feed_df = pd.DataFrame(st.session_state.manual_feed_data)
        st.dataframe(feed_df)


    if all(col in df.columns for col in ['date', 'pond_id', metric]):
        # KPI Summary Cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg FCR", f"{df['fcr'].mean():.2f}")
        col2.metric("Avg GPD", f"{df['gpd'].mean():.2f}")
        col3.metric("Avg Survival Rate", f"{df['survival_rate'].mean():.1%}")

        # Filter by pond
        filtered_df = df[df['pond_id'].isin(pond_filter)].copy()
        if smoothing:
            filtered_df['rolling'] = filtered_df.groupby('pond_id')[metric].transform(lambda x: x.rolling(7).mean())

        fig, ax = plt.subplots()
        for pond in pond_filter:
            subset = filtered_df[filtered_df['pond_id'] == pond]
            ax.plot(subset['date'], subset[metric], alpha=0.3, label=f"{pond} Raw")
            if smoothing:
                ax.plot(subset['date'], subset['rolling'], label=f"{pond} 7d Avg")
        if metric == 'fcr':
            ax.axhline(y=1.5, color='red', linestyle='--', label='FCR Benchmark')
        ax.set_title(f"{metric.upper()} Over Time")
        ax.legend()
        st.pyplot(fig)

    # PDF Export
    fig, ax = plt.subplots()
    for pond in df['pond_id'].unique():
        subset = df[df['pond_id'] == pond]
        ax.plot(subset['date'], subset[metric], label=pond)
    ax.set_title(f"{metric.upper()} Over Time")
    ax.legend()
    fig.tight_layout()
    pdf_buffer = BytesIO()
    fig.savefig(pdf_buffer, format='png')
    pdf_buffer.seek(0)

    
from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
img_path = "temp_plot.png"
with open(img_path, "wb") as f:
    f.write(pdf_buffer.read())
pdf.image(img_path, x=10, y=10, w=180)  # Use `pdf`, not `df`
pdf.image("logo.png", x=10, y=8, w=30)
pdf.ln(20)  # Add spacing after the logo

pdf_output_path = f"trend_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
pdf.output(pdf_output_path)

with open(pdf_output_path, "rb") as f:
    st.download_button("📥 Download Trend PDF", f, file_name=pdf_output_path, mime="application/pdf")
os.remove(img_path)

with tab2:
    st.subheader("📈 Correlation Matrix")
    features = ['avg_weight', 'survival_rate', 'fcr', 'gpd', 'do', 'saturation', 'nh3', 'nitrate', 'nitrite']
    valid = [col for col in features if col in df.columns]
    if valid:
        corr = df[valid].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Additional correlation visualizations
        st.subheader("Bar Chart of Mean KPI Values")
        mean_kpis = df[valid].mean()
        st.bar_chart(mean_kpis)

        st.subheader("Histograms of KPI Distributions")
        for feature in valid:
            fig, ax = plt.subplots()
            sns.histplot(df[feature].dropna(), bins=20, kde=True, ax=ax)
            ax.set_title(f"Histogram of {feature}")
            st.pyplot(fig)

        st.subheader("Boxplots of KPIs")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df[valid], ax=ax)
        ax.set_title("Boxplot of KPIs")
        st.pyplot(fig)


        # PDF Export
        pdf_buffer = BytesIO()
        fig.savefig(pdf_buffer, format='png')
        pdf_buffer.seek(0)

        pdf = FPDF()
        pdf.add_page()
        img_path = "temp_corr_plot.png"
        with open(img_path, "wb") as f:
            f.write(pdf_buffer.read())
        pdf.image(img_path, x=10, y=10, w=180)

        corr_pdf_path = f"correlation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(corr_pdf_path)

        with open(corr_pdf_path, "rb") as f:
            st.download_button("📥 Download Correlation PDF", f, file_name=corr_pdf_path, mime="application/pdf")
        os.remove(img_path)
        pdf.image("logo.png", x=10, y=8, w=30)
        pdf.ln(20)  # spacing after logo


    with tab3:
        st.subheader("🩺 Health Flags")
        fcr_thresh = st.number_input("FCR Threshold", value=2.0)
        gpd_thresh = st.number_input("GPD Threshold", value=1.0)
        do_thresh = st.number_input("DO Threshold", value=4.0)

        if 'fcr' in df.columns:
            df['fcr_flag'] = df['fcr'].apply(lambda x: 'High' if x > fcr_thresh else 'OK')
        if 'gpd' in df.columns:
            df['gpd_flag'] = df['gpd'].apply(lambda x: 'Low' if x < gpd_thresh else 'OK')
        if 'do' in df.columns:
            df['do_flag'] = df['do'].apply(lambda x: 'Low' if x < do_thresh else 'OK')

        flags = ['fcr_flag', 'gpd_flag', 'do_flag']
        st.dataframe(df[['pond_id', 'date'] + flags])
        display_cols = ['pond_id', 'date'] + flags
        display_cols = [col for col in display_cols if col in df.columns]
        if display_cols:
            colored_df = df[display_cols].copy()
            for flag in flags:
                colored_df[flag] = colored_df[flag].apply(
                    lambda x: f"🔴 {x}" if x == 'Low' else f"🟡 {x}" if x == 'High' else f"🟢 {x}"
                )
            st.dataframe(colored_df)


        # PDF Export
        fig, ax = plt.subplots(figsize=(10, 4))
        df['flag_count'] = df[flags].apply(lambda x: sum([f != 'OK' for f in x]), axis=1)
        df.groupby('date')['flag_count'].sum().plot(kind='bar', ax=ax)
        ax.set_title("Health Flags Over Time")
        ax.set_ylabel("Flag Count")
        fig.tight_layout()

        pdf_buffer = BytesIO()
        fig.savefig(pdf_buffer, format='png')
        pdf_buffer.seek(0)

        pdf = FPDF()
        pdf.add_page()
        img_path = "temp_health_plot.png"
        with open(img_path, "wb") as f:
            f.write(pdf_buffer.read())
        pdf.image(img_path, x=10, y=10, w=180)


        health_pdf_path = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(health_pdf_path)

        with open(health_pdf_path, "rb") as f:
            st.download_button("📥 Download Health Report (PDF)", f, file_name=health_pdf_path, mime="application/pdf")
        os.remove(img_path)
        pdf.image("logo.png", x=10, y=8, w=30)
        pdf.ln(20)  # spacing after logo



    with tab4:
        st.subheader("📋 Summary by Date")
        summary = df.groupby('date').agg({
            'fcr': 'mean', 'gpd': 'mean', 'do': 'mean',
            'survival_rate': 'mean', 'avg_weight': 'mean'
        }).reset_index()
        st.dataframe(summary)

        # CSV Export
        csv_data = summary.to_csv(index=False).encode('utf-8')
        st.download_button("📤 Download Summary (CSV)", csv_data, "summary.csv", mime="text/csv")
        pdf.image("logo.png", x=10, y=8, w=30)
        pdf.ln(20)  # spacing after logo


        # PDF Export
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Pond Performance Summary", ln=True)

        pdf.image("logo.png", x=10, y=8, w=30)
        pdf.ln(20)  # spacing after logo


        for i in range(len(summary)):
            row = summary.iloc[i]
            row_text = ", ".join([f"{col}: {row[col]:.2f}" if isinstance(row[col], float) else f"{col}: {row[col]}" for col in summary.columns])
            pdf.cell(200, 8, txt=row_text, ln=True)

        summary_pdf_path = f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(summary_pdf_path)

        with open(summary_pdf_path, "rb") as f:
            st.download_button("📥 Download Summary (PDF)", f, file_name=summary_pdf_path, mime="application/pdf")
    
    with tab5:
        st.subheader("\U0001F489 Fish Disease Detection")
        st.write("Upload a fish image and enter observed symptoms to detect possible diseases.")
    
        uploaded_image = st.file_uploader("Upload Fish Image", type=['jpg', 'jpeg', 'png'])
        symptoms = st.text_area("Observed Signs/Symptoms", placeholder="e.g., red spots, swollen eyes, frayed fins")
    
        if st.button("Analyze Disease"):
            if uploaded_image and symptoms:
                try:
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Uploaded Fish Image", use_column_width=True)
                    # Placeholder prediction logic
                    possible_diseases = [
                        "Bacterial Infection",
                        "Parasitic Infestation",
                        "Fungal Growth",
                        "Nutritional Deficiency"
                    ]
                    st.success(f"Based on signs: {symptoms}, possible diseases include:")
                    for disease in possible_diseases:
                        st.markdown(f"- {disease}")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
            else:
                st.warning("Please upload an image and enter symptoms to analyze.")





