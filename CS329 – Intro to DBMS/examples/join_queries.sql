-- join_queries.sql
-- INNER JOIN
SELECT p.id, p.title, u.username
FROM posts p
JOIN users u ON p.user_id = u.id;

-- LEFT JOIN
SELECT u.username, p.title
FROM users u
LEFT JOIN posts p ON p.user_id = u.id;
