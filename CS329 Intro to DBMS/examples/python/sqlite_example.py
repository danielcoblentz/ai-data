"""sqlite_example.py
Simple example using the builtin sqlite3 module.
Creates a database file (example.db) from the SQL schema and runs a couple of queries.
"""
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent
DB = BASE / '..' / 'db' / 'example.db'
DB.parent.mkdir(parents=True, exist_ok=True)

schema_file = BASE.parent / 'sql' / 'create_schema.sql'

conn = sqlite3.connect(DB)
cur = conn.cursor()

with open(schema_file, 'r', encoding='utf-8') as f:
    schema = f.read()
cur.executescript(schema)
conn.commit()

print('Database created at', DB)

cur.execute('SELECT id, username, email FROM users')
rows = cur.fetchall()
print('Users:')
for r in rows:
    print(r)

cur.execute('SELECT id, title, user_id FROM posts')
rows = cur.fetchall()
print('\nPosts:')
for r in rows:
    print(r)

conn.close()
