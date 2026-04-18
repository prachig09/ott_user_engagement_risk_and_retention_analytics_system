import gradio as gr
from ui.styles import CSS
from ui.components.sidebar import render_sidebar
from ui.pages.home import render_home_page
from ui.pages.predict import render_predict_page
from ui.pages.upload import render_upload_page
from ui.pages.reports import render_reports_page

# 1. Custom Color Palette
ott_green = gr.themes.Color(
    name="ott_green",
    c50="#e0f7f1", c100="#b2e0d4", c200="#66b3a1", c300="#4da691",
    c400="#26927a", c500="#007a33", c600="#3a3b3a", c700="#004d00",
    c800="#000000", c900="#000000", c950="#001f00",
)

# 2. Theme Initialization
theme = gr.themes.Soft(
    primary_hue=ott_green, 
    secondary_hue=ott_green,
    neutral_hue="slate",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
    # FIX: Ensure the soft theme doesn't override your sidebar CSS
    body_background_fill="white",
)

# 3. Application Layout
# FIX: Use 'head' parameter here too as a secondary backup for mounting
with gr.Blocks(
    title="OTT Retention System", 
    css=CSS, 
    theme=theme,
    delete_cache=(60, 3600),
    head=f"<style>{CSS}</style>" 
) as demo:
    
    active_tab = gr.State(0)

    with gr.Row(equal_height=False): # Add equal_height=False to prevent sidebar stretching weirdly
        # Render Sidebar (Ensure ui/components/sidebar.py has elem_classes="sidebar-panel")
        nav_home, nav_upload, nav_predict, nav_reports = render_sidebar()
        
        with gr.Column(scale=4):
            with gr.Tabs() as tabs:
                with gr.Tab("Home", id=0):
                    render_home_page()
                
                with gr.Tab("Upload", id=1):
                    render_upload_page()
                
                with gr.Tab("Predict", id=2):
                    render_predict_page()
                
                with gr.Tab("Reports", id=3):
                    render_reports_page()
            
    # Navigation Logic
    nav_home.click(fn=lambda: gr.Tabs(selected=0), outputs=tabs) 
    nav_upload.click(fn=lambda: gr.Tabs(selected=1), outputs=tabs)
    nav_predict.click(fn=lambda: gr.Tabs(selected=2), outputs=tabs)
    nav_reports.click(fn=lambda: gr.Tabs(selected=3), outputs=tabs)

# DO NOT put demo.launch() outside the if __name__ == "__main__" block
# This prevents it from accidentally starting a separate server when imported by run_app.py
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        show_error=True,
        quiet=True
    )