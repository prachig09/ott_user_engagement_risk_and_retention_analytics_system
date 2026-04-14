import gradio as gr
from ui.styles import CSS
from ui.components.sidebar import render_sidebar
from ui.pages.home import render_home_page
from ui.pages.predict import render_predict_page
from ui.pages.upload import render_upload_page
from ui.pages.reports import render_reports_page

# 1. Define the custom color palette correctly to avoid the ValueError
# This maps your grounding and light shades to the Gradio scale
ott_green = gr.themes.Color(
    name="ott_green",
    c50="#e0f7f1",    # Lightest shade
    c100="#b2e0d4",   # Light shade
    c200="#66b3a1",   # Light accent
    c300="#4da691",
    c400="#26927a",
    c500="#007a33",   # Main grounding green
    c600="#3a3b3a",
    c700="#004d00",   # Deep grounding green
    c800="#000000",
    c900="#000000",
    c950="#001f00",
)

# 2. Initialize the Glass theme with the Color object
# We use the ott_green for primary (buttons/links) and secondary (accents)
theme = gr.themes.Soft(
    primary_hue=ott_green, 
    secondary_hue=ott_green,
    neutral_hue="slate", # Professional grey for text and backgrounds
).set(
    # Optional: Force specific grounding colors for primary buttons
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
)
active_tab = gr.State(0)
# 3. Application Layout
with gr.Blocks(title="OTT Retention System", css=CSS, theme=theme) as demo:
    with gr.Row():
        # Render Sidebar
        # Returns: home, upload, predict, reports
        nav_home, nav_upload, nav_predict, nav_reports = render_sidebar()
        
        # Main Content Area
        with gr.Column(scale=4):
            with gr.Tabs() as tabs:
                with gr.Tab("Home", id=0):
                    render_home_page()
                
                with gr.Tab("Upload", id=1):
                    render_upload_page()
                
                #with gr.Tab("Predict", id=2):
                    #render_predict_page()
                
                #with gr.Tab("Reports", id=3):
                    #render_reports_page()
            
    # Connect Sidebar buttons to Tabs using gr.Tabs.update (Gradio 4.x style)
    nav_home.click(fn=lambda: gr.update(selected=0), outputs=tabs) # Home
    nav_upload.click(fn=lambda: gr.update(selected=1), outputs=tabs) # Upload
    nav_predict.click(fn=lambda: gr.update(selected=2), outputs=tabs) # Predict
    nav_reports.click(fn=lambda: gr.update(selected=3), outputs=tabs) # Reports

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        show_api=False,  # This stops the 'get_api_info' crash!
        show_error=True
    )