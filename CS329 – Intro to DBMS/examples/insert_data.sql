-- insert_data.sql
INSERT INTO users (username, email) VALUES ('alice', 'alice@example.com');
INSERT INTO users (username, email) VALUES ('bob', 'bob@example.com');

INSERT INTO posts (user_id, title, body) VALUES (1, 'Hello', 'First post');
INSERT INTO posts (user_id, title, body) VALUES (2, 'Bob''s post', 'Hi from Bob');

INSERT INTO comments (post_id, author, body) VALUES (1, 'charlie', 'Nice post');
