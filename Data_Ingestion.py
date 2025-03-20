import pandas as pd

def load_and_clean_plant_data(file_path: str) -> pd.DataFrame:
    """
    Reads raw plant CSV data, sets row 1 as the column headers,
    removes the placeholder rows, and sanitizes the Timestamp column.
    """
    # 1. Read the raw file with the specified encoding
    raw_df = pd.read_csv(file_path, encoding='latin-1')
    
    # 2. Extract row 1 to use as the actual headers
    raw_df.columns = raw_df.iloc[1]
    
    # 3. Drop rows 0 and 1 (keeping everything from index 2 onwards)
    # and reset the index so it starts fresh at 0
    cleaned_df = raw_df.iloc[2:].reset_index(drop=True)
    
    # 4. Clear out the axis label name if it gets stuck with '1'
    cleaned_df.columns.name = None
    
    # 5. Rename the first column explicitly to 'Timestamp'
    cleaned_df.columns.values[0] = 'Timestamp'
    
    # 6. Format the Timestamp column, coercing errors to NaT
    cleaned_df['Timestamp'] = pd.to_datetime(cleaned_df['Timestamp'], errors='coerce')
    
    return cleaned_df
