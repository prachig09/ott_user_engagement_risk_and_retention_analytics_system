import gradio as gr
from styles import CSS
from components.sidebar import render_sidebar
from pages.home import render_home_page
from pages.predict import render_predict_page
from pages.upload import render_upload_page
from pages.reports import render_reports_page


theme=gr.themes.Citrus(
    primary_hue="red", 
    secondary_hue="gray",
)

# Default theme (no special theme passed)
with gr.Blocks(title="OTT Retention System",css=CSS,theme=theme) as demo:
    with gr.Row():
        # Render Sidebar
        # Returns: home, upload, predict, reports, settings
        nav_btns = render_sidebar()
        
        # Main Content Area
        with gr.Column(scale=4):
            with gr.Tabs() as tabs:
                with gr.TabItem("Home", id=0):
                    render_home_page()
                
                with gr.TabItem("Upload", id=1):
                    render_upload_page()
                
                with gr.TabItem("Predict", id=2):
                    render_predict_page()
                
                with gr.TabItem("Reports", id=3):
                    render_reports_page()
            

    # Connect Sidebar buttons to Tabs using gr.update
    nav_btns[0].click(fn=lambda: gr.update(selected=0), outputs=tabs) # Home
    nav_btns[1].click(fn=lambda: gr.update(selected=1), outputs=tabs) # Upload
    nav_btns[2].click(fn=lambda: gr.update(selected=2), outputs=tabs) # Predict
    nav_btns[3].click(fn=lambda: gr.update(selected=3), outputs=tabs) # Reports
if __name__ == "__main__":
    demo.launch()