import pandas as pd
import sqlite3
import glob
import os

def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, 'generated')
    db_path = os.path.join(base_dir, 'eduquality.db')

    # Create sqlite connection
    conn = sqlite3.connect(db_path)
    
    # Get all csv files
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return

    print(f"Found {len(csv_files)} CSV files. Creating SQLite DB at {db_path}...")

    for csv_file in csv_files:
        # Extract filename without extension for table name
        table_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Write to SQLite
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"Successfully loaded {table_name} table with {len(df)} rows.")
        except Exception as e:
            print(f"Failed to load {csv_file}: {e}")

    # Create a mock subjects assignment table
    try:
        # Create a mock table that assigns subjects to a specific teacher
        # From the student records we know which subjects exist (ma_mon)
        df_students = pd.read_csv(os.path.join(csv_dir, 'students.csv'))
        unique_subjects = df_students['ma_mon'].unique()
        
        # We assign all subjects to teacher 'GV001' for demo
        teacher_subject_df = pd.DataFrame({
            'teacher_id': ['GV001'] * len(unique_subjects),
            'ma_mon': unique_subjects
        })
        teacher_subject_df.to_sql('teacher_subjects', conn, if_exists='replace', index=False)
        print("Successfully created teacher_subjects mock table.")
        
    except Exception as e:
        print(f"Could not create mock teacher_subjects table: {e}")

    conn.close()
    print("Database creation complete.")

if __name__ == "__main__":
    main()
