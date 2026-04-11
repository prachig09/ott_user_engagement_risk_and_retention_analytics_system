import gradio as gr
from components.sidebar import render_sidebar
from pages.home import render_home_page
from pages.predict import render_predict_page
# Import other pages similarly...

with gr.Blocks(title="OTT Retention System") as demo:
    with gr.Row():
        # Render Sidebar
        nav_btns = render_sidebar()
        
        # Main Content Area
        with gr.Column(scale=4):
            with gr.Tabs() as tabs:
                with gr.TabItem("Home", id=0):
                    home_page = render_home_page()
                
                with gr.TabItem("Upload", id=1):
                    gr.Markdown("# Upload Page Content")
                
                with gr.TabItem("Predict", id=2):
                    predict_page = render_predict_page()

    # Connect Sidebar buttons to Tabs
    nav_btns[0].click(fn=lambda: gr.Tabs(selected=0), outputs=tabs)
    nav_btns[1].click(fn=lambda: gr.Tabs(selected=1), outputs=tabs)
    nav_btns[2].click(fn=lambda: gr.Tabs(selected=2), outputs=tabs)

if __name__ == "__main__":
    demo.launch()