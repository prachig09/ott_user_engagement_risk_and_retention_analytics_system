import gradio as gr
import pandas as pd
import plotly.express as px
from components.widgets import metric_card

def get_churn_distribution_plot():
    # Synthetic data for the dashboard overview
    data = pd.DataFrame({
        "Risk Level": ["Low Risk", "Moderate Risk", "High Risk"],
        "Customer Count": [450, 300, 150]
    })
    
    fig = px.pie(
        data, 
        values='Customer Count', 
        names='Risk Level',
        title="Overall Customer Risk Distribution",
        color='Risk Level',
        color_discrete_map={
            'Low Risk': '#2ecc71',
            'Moderate Risk': '#f1c40f',
            'High Risk': '#e74c3c'
        }
    )
    
    # Update layout for better UI fit
    fig.update_layout(
        margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.2)
    )
    
    return fig

def get_feature_importance_plot():
    # These represent the 'Global Drivers' of churn from your model training
    importance_data = pd.DataFrame({
        "Feature": ["Last Login Days", "Payment Failures", "Support Calls", "Watch Time", "Monthly Charges"],
        "Impact Score": [0.85, 0.72, 0.65, 0.45, 0.30]
    }).sort_values(by="Impact Score", ascending=True)

    fig = px.bar(
        importance_data, x="Impact Score", y="Feature", 
        orientation='h', 
        title="Top Drivers of Churn",
        color_discrete_sequence=['#0984e3']
    )
    fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=400)
    return fig

def render_home_page():
    with gr.Column() as page:
        gr.Markdown("# 📊 Dashboard")
        gr.Markdown("Welcome! Here is the current status of your OTT platform engagement.")

        # Metrics Row - using your metric_card widget
        with gr.Row():
            metric_card("Total Customers", "1,240", "#0984e3")
            metric_card("Churn Rate", "14.2%", "#d63031")
            metric_card("Model Accuracy", "91.5%", "#00b894")
            metric_card("High Risk Users", "28", "#fdcb6e")

        # Visuals Row
        with gr.Row():
            with gr.Column(scale=2,):
                gr.Markdown("### 🛠️ Model Status")
                gr.Markdown("""
                - **Algorithm:** Gradient Boosting
                - **Precision:** 0.89 
                - **Recall:** 0.84
                - **Last Retrained:** 2 days ago
                """)
                gr.Markdown("### 💡 Key Insights (Global Drivers)")
                gr.Plot(value=get_feature_importance_plot(), show_label=False)
                
            with gr.Column(scale=3):
                gr.Markdown("### 🥧 Risk Segmentation")
                # Passing the fig directly to the Plot component
                gr.Plot(value=get_churn_distribution_plot(), show_label=False)
                
    return page