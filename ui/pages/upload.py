import gradio as gr
import pandas as pd
import requests
import plotly.express as px

BATCH_API_URL = "http://0.0.0.0:8000/predict_batch"

def create_visuals(df):
    if df is None or df.empty:
        return None, None
    
    # 🥧 1. Risk Distribution Pie Chart
    risk_counts = df['Risk_Level'].value_counts().reset_index()
    risk_counts.columns = ['Risk Level', 'Count']
    fig_pie = px.pie(
        risk_counts, 
        values='Count', 
        names='Risk Level', 
        title="Batch Risk Distribution",
        color_discrete_map={'HIGH': '#e74c3c', 'MODERATE': '#f1c40f', 'LOW': '#2ecc71'}
    )
    
    # 📊 2. Probability Histogram
    fig_hist = px.histogram(
        df, 
        x="Churn_Probability", 
        nbins=20, 
        title="Churn Probability Spread",
        color_discrete_sequence=['#3498db']
    )
    fig_hist.update_layout(bargap=0.1)
    
    return fig_pie, fig_hist

def run_batch_analysis(file):
    if file is None:
        return None, "❌ No file uploaded.", None, None
    
    try:
        df = pd.read_csv(file.name)
        records = df.to_dict(orient="records")
        payload = {"data": records}
        
        response = requests.post(BATCH_API_URL, json=payload, timeout=120)
        api_results = response.json()
        
        if api_results.get("status") == "success":
            results_df = pd.DataFrame(api_results["predictions"])
            
            # Generate the charts based on new data
            pie_chart, hist_chart = create_visuals(results_df)
            
            return results_df, f"✅ Analyzed {len(results_df)} records.", pie_chart, hist_chart
        else:
            return None, f"❌ API Error: {api_results.get('message')}", None, None
            
    except Exception as e:
        return None, f"❌ Connection Error: {str(e)}", None, None

def render_upload_page():
    with gr.Column() as page:
        gr.Markdown("# 📂 Batch Processing & Insights")
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload CSV", file_types=[".csv"])
                process_btn = gr.Button("🚀 Run Batch Analysis", variant="primary")
                status_msg = gr.Markdown("Upload a file to begin.")
            
            # 📈 Top Visuals Area
            with gr.Column(scale=2):
                with gr.Row():
                    pie_plot = gr.Plot(label="Risk Summary")
                    hist_plot = gr.Plot(label="Probability Spread")

        gr.Markdown("---")
        
        # 📄 Table Area
        with gr.Column():
            gr.Markdown("### 📋 Detailed Results")
            results_table = gr.DataFrame(interactive=False)

        # Event Logic
        process_btn.click(
            fn=run_batch_analysis,
            inputs=[file_input],
            outputs=[results_table, status_msg, pie_plot, hist_plot]
        )

    return page