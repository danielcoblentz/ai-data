-- create_schema.sql
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS posts;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- sample data
INSERT INTO users (username, email) VALUES ('alice', 'alice@example.com');
INSERT INTO users (username, email) VALUES ('bob', 'bob@example.com');
INSERT INTO users (username, email) VALUES ('carol', 'carol@example.com');

INSERT INTO posts (user_id, title, body) VALUES (1, 'Hello world', 'First post by Alice');
INSERT INTO posts (user_id, title, body) VALUES (1, 'Another post', 'Alice again');
INSERT INTO posts (user_id, title, body) VALUES (2, 'Bob''s post', 'Post by Bob');
INSERT INTO posts (user_id, title, body) VALUES (3, 'Carol says hi', 'A hello from Carol');
