import sqlite3

# Define the database name
db_name = "results.db"

# Connect to SQLite (or create the database if it doesn't exist)
conn = sqlite3.connect(db_name)

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Define the SQL command to create the table
create_table_query = """
CREATE TABLE IF NOT EXISTS best_results (
    instance TEXT,
    subset TEXT,
    method TEXT,
    seed INTEGER,
    steps INTEGER,
    initial_objective_value REAL,
    objective_value REAL,
    time REAL,
    avg_time_per_step REAL
);
"""

# Execute the command to create the table
cursor.execute(create_table_query)

# Commit the changes and close the connection
conn.commit()
conn.close()

print(f"Database and table created successfully in '{db_name}'!")
