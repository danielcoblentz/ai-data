-- queries.sql: assorted example queries

-- 1) Basic SELECT
SELECT id, username, email FROM users;

-- 2) WHERE, comparison
SELECT * FROM posts WHERE user_id = 1;

-- 3) LIKE and pattern matching
SELECT username FROM users WHERE username LIKE 'a%';

-- 4) DISTINCT
SELECT DISTINCT user_id FROM posts;

-- 5) GROUP BY / aggregate
SELECT user_id, COUNT(*) AS post_count FROM posts GROUP BY user_id;

-- 6) HAVING (filter aggregates)
SELECT user_id, COUNT(*) AS post_count FROM posts GROUP BY user_id HAVING COUNT(*) > 1;

-- 7) ORDER BY / LIMIT
SELECT * FROM posts ORDER BY created_at DESC LIMIT 3;

-- 8) INNER JOIN
SELECT p.id, p.title, u.username
FROM posts p
JOIN users u ON p.user_id = u.id;

-- 9) LEFT JOIN (preserve users with no posts)
SELECT u.id, u.username, p.title
FROM users u
LEFT JOIN posts p ON p.user_id = u.id;

-- 10) Subquery (IN)
SELECT username FROM users WHERE id IN (SELECT DISTINCT user_id FROM posts);

-- 11) EXISTS
SELECT username FROM users u WHERE EXISTS (SELECT 1 FROM posts p WHERE p.user_id = u.id);

-- 12) CTE (WITH)
WITH recent_posts AS (
  SELECT * FROM posts ORDER BY created_at DESC LIMIT 5
)
SELECT rp.id, rp.title, u.username FROM recent_posts rp JOIN users u ON rp.user_id = u.id;

-- 13) Window function (SQLite supports ROW_NUMBER from v3.25+)
SELECT id, title, user_id,
  ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) AS rn
FROM posts;

-- 14) Transaction example (illustrative; runner executes statements separately)
BEGIN TRANSACTION;
INSERT INTO users (username, email) VALUES ('dave', 'dave@example.com');
INSERT INTO posts (user_id, title, body) VALUES (4, 'Dave''s first post', 'Welcome Dave');
COMMIT;

-- 15) Create an index for speed
CREATE INDEX IF NOT EXISTS idx_posts_user ON posts(user_id);

-- 16) Delete
DELETE FROM posts WHERE id = -1; -- no-op example
