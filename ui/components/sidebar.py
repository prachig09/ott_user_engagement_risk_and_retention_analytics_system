import gradio as gr

def render_sidebar():
    with gr.Column(scale=1, variant="compact"):
        gr.Markdown("## 🧭 ChurnGuard")
        
        # Standard buttons without group or special classes
        nav_home = gr.Button("🏠 Home")
        nav_upload = gr.Button("📂 Batch Upload")
        nav_predict = gr.Button("🔍 Predict Risk")
        nav_reports = gr.Button("📄 Reports")

        gr.Markdown("<br>" * 5)
        gr.Markdown("---") # Replaces the Divider
        gr.Markdown("Model Version: **v1.0.4**")
        
    return nav_home, nav_upload, nav_predict, nav_reports