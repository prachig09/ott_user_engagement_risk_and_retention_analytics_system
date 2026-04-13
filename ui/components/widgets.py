import gradio as gr

def metric_card(title, value, color="#2d3436"):
    """Creates a stylized metric card using HTML."""
    html_content = f"""
    <div style='background-color: white; padding: 20px; border-radius: 12px; 
                border-left: 8px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <p style='margin: 0; color: #636e72; font-size: 14px; font-weight: bold;'>{title}</p>
        <h2 style='margin: 5px 0 0 0; color: #2d3436; font-size: 28px;'>{value}</h2>
    </div>
    """
    return gr.HTML(html_content)