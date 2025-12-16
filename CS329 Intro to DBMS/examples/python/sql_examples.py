"""sql_examples.py
Runner that executes statements from examples/sql/queries.sql against the sqlite DB created by sqlite_example.py
"""
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent
DB = BASE / '..' / 'db' / 'example.db'
SQL_FILE = BASE.parent / 'sql' / 'queries.sql'

if not DB.exists():
    print('Database not found at', DB)
    print('Run sqlite_example.py first to create the example DB')
    raise SystemExit(1)

sql_text = SQL_FILE.read_text(encoding='utf-8')
# naive split by ; for demo purposes
stmts = [s.strip() for s in sql_text.split(';') if s.strip()]

conn = sqlite3.connect(DB)
cur = conn.cursor()

for i, stmt in enumerate(stmts, 1):
    try:
        print(f'----- Statement {i} -----')
        print(stmt[:200])
        cur.execute(stmt)
        if stmt.lstrip().upper().startswith('SELECT'):
            rows = cur.fetchmany(5)
            cols = [d[0] for d in cur.description] if cur.description else []
            print('Columns:', cols)
            for r in rows:
                print(r)
        else:
            conn.commit()
            print('OK')
    except Exception as e:
        print('ERROR:', e)

conn.close()
