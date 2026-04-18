import gradio as gr

def render_sidebar():
    # We add elem_classes here so it matches the .sidebar-panel in your CSS
    with gr.Column(scale=1, variant="compact", elem_classes="sidebar-panel"):
        gr.Markdown("# 🧭 IVAS")
        gr.Markdown("### Ingest • Validate • Automate • Scale")
        
        nav_home = gr.Button(" Home")
        nav_upload = gr.Button(" Batch Upload")
        nav_predict = gr.Button(" Predict Risk")
        nav_reports = gr.Button(" Reports")

        gr.Markdown("<br>" * 5)
        gr.Markdown("---") 
        gr.Markdown("Model Version: **v1.0.4**")
        
    return nav_home, nav_upload, nav_predict, nav_reports