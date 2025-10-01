CS329 — Intro to DBMS

This folder contains introductory teaching materials and runnable examples using SQLite and SQLAlchemy.

Files added:

- examples/sql/create_schema.sql  — schema + sample data
- examples/sql/queries.sql       — many example SQL queries (SELECT, JOINs, GROUP BY, WINDOW, CTE, transactions)
- examples/python/sqlite_example.py   — simple sqlite3 example: create DB and query
- examples/python/sqlalchemy_example.py — SQLAlchemy declarative example
- examples/python/sql_examples.py  — runner that executes statements from queries.sql against the SQLite DB
- examples/python/requirements.txt  — minimal Python requirements

How to run the examples (PowerShell):

# create the sqlite DB and run the SQLAlchemy example
cd "c:\Users\dan\Desktop\ai-data\CS329 – Intro to DBMS\examples\python"
python sqlalchemy_example.py

# run the SQL files using the runner
python sql_examples.py

Notes:
- These examples use SQLite for simplicity so they should run without installing a database server.
- If you'd like Postgres/Docker examples, say so and I'll add them.
