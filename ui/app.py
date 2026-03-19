"""
Customer Churn Prediction Application - Main Entry Point
Modern dashboard UI with sidebar navigation and multiple pages.

"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# Add ui directory to path
ui_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ui_dir)

from theme import COLORS, FONTS, SIZES
from components.sidebar import Sidebar
from pages.home import HomePage
from pages.predict import PredictPage
from pages.upload import UploadPage
from pages.upload_result import UploadResultPage
from pages.charts import ChartsPage
from pages.reports import ReportsPage
from pages.settings import SettingsPage


class ChurnPredictionApp:
    """Main application class with sidebar navigation."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AI Customer Churn Prediction System")
        self.root.geometry("1300x800")
        self.root.minsize(1100, 700)
        self.root.configure(bg=COLORS['bg_dark'])
        
        # Current page tracking
        self.current_page = None
        self.pages = {}
        
        # Setup UI
        self.setup_ui()
        
        # Show home page by default
        self.show_page('home')
    
    def setup_ui(self):
        """Setup the main UI layout."""
        # Main container
        main_container = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Sidebar
        self.sidebar = Sidebar(
            main_container,
            on_page_change=self.show_page,
            width=SIZES['sidebar_width']
        )
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        self.sidebar.config(width=SIZES['sidebar_width'])
        
        # Content area
        self.content_area = tk.Frame(main_container, bg=COLORS['bg_medium'])
        self.content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create all pages
        self.create_pages()
    
    def create_pages(self):
        """Create all page instances."""
        page_classes = {
            'home': HomePage,
            'predict': PredictPage,
            'upload': UploadPage,
            'upload_result': UploadResultPage,
            'charts': ChartsPage,
            'report': ReportsPage,
            'settings': SettingsPage,
        }
        
        for page_id, page_class in page_classes.items():
            page = page_class(self.content_area, controller=self)
            self.pages[page_id] = page
    
    def show_page(self, page_id):
        """Show the specified page."""
        # Hide current page
        if self.current_page and self.current_page in self.pages:
            self.pages[self.current_page].pack_forget()
            if hasattr(self.pages[self.current_page], 'on_hide'):
                self.pages[self.current_page].on_hide()
        
        # Show new page
        if page_id in self.pages:
            self.pages[page_id].pack(fill=tk.BOTH, expand=True)
            self.current_page = page_id
            
            # Update sidebar active state
            self.sidebar.set_active(page_id)
            
            # Trigger page show callback
            if hasattr(self.pages[page_id], 'on_show'):
                self.pages[page_id].on_show()
        else:
            messagebox.showwarning("Page Not Found", f"Page '{page_id}' not found.")


def main():
    """Run the application."""
    root = tk.Tk()
    
    # Set window icon
    try:
        # You can set an icon here if available
        pass
    except:
        pass
    
    # Create and run app
    app = ChurnPredictionApp(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()