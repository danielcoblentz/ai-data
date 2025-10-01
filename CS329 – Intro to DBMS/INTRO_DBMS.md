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


# run the SQL files using the runner
python sql_examples.py

