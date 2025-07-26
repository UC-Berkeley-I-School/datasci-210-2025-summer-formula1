import psycopg2

try:
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=5432,
        database="racing_telemetry",
        user="racing_user",
        password="racing_password"
    )
    print("Connected to PostgreSQL successfully!")
    conn.close()
except Exception as e:
    print("Failed to connect to PostgreSQL:", e)
