CSS = """
/* --- 1. GLOBAL CONTAINER FIXES --- */
.gradio-container {
    max-width: 100% !important; 
    margin: 0 !important;
    padding: 0 !important;
}

/* --- 2. THE SIDEBAR --- */
/* We use Gradio's internal surface color variables so it flips in dark mode */
.sidebar-panel {
   background-color: var(--background-fill-secondary) !important;
   border-right: 1px solid var(--border-color-primary) !important;
   min-height: 100vh !important;
   padding: 20px !important;
   display: flex !important;
   flex-direction: column !important;
}

/* --- 3. TEXT & FONT SCALING --- */
/* Targeted to ensure it doesn't break icons or layout */
body, p, span, li, label, .prose, .prose p, .prose li {
    font-size: 18px !important;
    line-height: 1.6 !important;
}

/* --- 4. CARDS & UI COMPONENTS --- */
.metric-card {
   /* --block-background-fill is white in light mode, dark grey in dark mode */
   background-color: var(--block-background-fill) !important;
   padding: 15px !important;
   border-radius: 10px !important;
   border: 1px solid var(--border-color-primary) !important;
   text-align: center !important;
   box-shadow: var(--block-shadow) !important;
   color: var(--body-text-color) !important;
}

.status-card {
   /* Status cards usually stay dark, but we ensure text is readable */
   background-color: var(--neutral-800) !important;
   color: white !important;
   padding: 20px !important;
   border-radius: 10px !important;
}

/* --- 5. BUTTON OVERRIDES --- */
button.primary {
    background-color: var(--primary-600) !important;
    color: white !important;
    border: none !important;
}

button.secondary {
    background-color: var(--neutral-200) !important;
    color: var(--neutral-800) !important;
}

/* --- 6. DARK MODE SPECIFIC TWEAKS --- */
/* This ensures that if the system is in dark mode, neutral-200 buttons become dark */
@media (prefers-color-scheme: dark) {
    button.secondary {
        background-color: var(--neutral-700) !important;
        color: white !important;
    }
}
"""