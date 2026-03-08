from pathlib import Path
import pandas as pd

def parse_data():
    # Get the directory where this script lives
    script_dir = Path(__file__).parent
    file_path = script_dir / "data.xlsx"

    data = pd.read_excel(file_path)
    return data