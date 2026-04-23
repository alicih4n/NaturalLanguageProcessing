import subprocess
import sys
import os

def export_to_pdf():
    notebook_name = "PathoIntern_Intent_Engine.ipynb"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(script_dir, notebook_name)
    
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_name} not found in {script_dir}")
        return

    print(f"Exporting {notebook_name} to PDF...")
    
    try:
        # We use --to webpdf which uses a headless browser (like Playwright/Chromium)
        # to avoid complex LaTeX dependencies.
        subprocess.run([
            sys.executable, "-m", "jupyter", "nbconvert", 
            "--to", "webpdf", 
            "--allow-chromium-download",
            notebook_path
        ], check=True)
        print(f"Success! PDF created in {script_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    export_to_pdf()
