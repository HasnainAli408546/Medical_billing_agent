"""One-time cleanup script to remove duplicate page files."""
import os, shutil

pages_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")

# Remove the entire pages directory and its contents
if os.path.exists(pages_dir):
    shutil.rmtree(pages_dir, ignore_errors=True)
    print(f"Removed: {pages_dir}")

# Also remove old theme.py
theme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theme.py")
if os.path.exists(theme_path):
    os.remove(theme_path)
    print(f"Removed: {theme_path}")

print("Cleanup complete! Now run: streamlit run ui_app.py")
