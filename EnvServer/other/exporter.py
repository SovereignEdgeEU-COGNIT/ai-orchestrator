import pandas as pd
import psycopg2

# Specify your database connection details
conn_details = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'rFcLGNkgsNtksg6Pgtn9CumL4xXBQ7',
    'host': '192.168.1.156',
    'port': '5432'
}

# Define the SQL query
# Adjust the query if your 'id' column has a different name
sql_query = """
SELECT *
FROM prod_metrics
"""

#WHERE id NOT LIKE '%_'

# Specify your desired CSV file path
csv_file_path = 'exportedMetrics.csv'

# Specify the header names you want in your CSV file
# Replace 'column1', 'column2', ..., 'columnN' with your actual column names
csv_headers = ['id', 'column2', 'columnN']

try:
    # Connect to your database
    with psycopg2.connect(**conn_details) as conn:
        # Load the query results into a pandas DataFrame
        df = pd.read_sql_query(sql_query, conn)
        
        # If you want to rename the DataFrame columns to your specified headers
        #df.columns = csv_headers

        # Export the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)

    print("Data exported successfully to", csv_file_path)

except Exception as e:
    print("Error occurred:", e)
