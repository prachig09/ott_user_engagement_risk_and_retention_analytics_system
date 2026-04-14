# ui/styles.py


CSS = """

/* --- 1. RESPONSIVE & FLUID LAYOUT --- */
.gradio-container {
    max-width: 100% !important; 
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
}

/* --- 2. GLOBAL TEXT & FONT SIZES --- */
/* "Normal text" 18px */
body, p, span, li, label, .prose, .prose p, .prose li {
    font-size: 18px !important;
    line-height: 1.6 !important;
}
.sidebar-panel {
   background-color: #f8f9fa;
   border-right: 1px solid #ddd;
   min-height: 100vh;
}
.metric-card {
   background-color: white;
   padding: 15px;
   border-radius: 10px;
   border: 1px solid #e0e0e0;
   text-align: center;
   box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
}
.status-card {
   background-color: #2d3436;
   color: white;
   padding: 20px;
   border-radius: 10px;
}
"""
