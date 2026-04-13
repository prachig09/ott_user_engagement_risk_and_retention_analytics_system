import gradio as gr

def render_reports_page():
    with gr.Column() as page:
        gr.Markdown("# 📄 MLOps Artifacts & Experiment Reports")
        gr.Markdown("Monitoring model lineage via **DVC** and experiment tracking via **MLflow**.")

        with gr.Row():
            # --- MLflow Section ---
            with gr.Column(variant="panel"):
                gr.Markdown("### 🧪 MLflow Tracking")
                gr.Markdown("""
                **Current Status:** 🟢 Active  
                **Database:** `mlflow.db` (SQLite)  
                **Active Experiment:** `0 (Default)`
                
                Click below to view the full performance leaderboard, including:
                - Hyperparameter tuning (Learning Rate, Depth)
                - Accuracy/Loss curves
                - Model registration history
                """)
                mlflow_btn = gr.Button("🚀 Open MLflow Dashboard", variant="primary")

            # --- DVC Section ---
            with gr.Column(variant="panel"):
                gr.Markdown("### 🧬 Data Versioning (DVC)")
                gr.Markdown("""
                **Current Status:** 🔒 Versioned  
                **Remote Storage:** Local Cache  
                **Tracked Entities:** `data/`, `model/`
                
                DVC ensures that the data used for training is perfectly synced with the model binary deployed in this UI.
                """)
                gr.Image("assets/dvc_dag.png", label="Pipeline DAG")
                gr.Image("assets/dvc_dag_ss.png", label="Pipeline DAG Terminal Screenshot")
        # --- System Health / Audit Log ---
        gr.Markdown("### 🛠️ System Metadata")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Version Control**")
                gr.Code(value="Git: v1.0.4-beta\nDVC: v3.x", language="markdown")
            with gr.Column(scale=1):
                gr.Markdown("**Environment**")
                gr.Code(value="Python: 3.9+\nFramework: Gradio", language="markdown")

        # --- Button Logic (Browser Redirects) ---
        # Note: MLflow usually runs on port 5000. 
        # For DVC, we link to the DAG documentation or your local remote.
        mlflow_btn.click(None, None, None, js='() => { window.open("http://127.0.0.1:5000", "_blank"); }')
    return page