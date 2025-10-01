-- select_queries.sql
SELECT * FROM users;
SELECT id, title FROM posts WHERE user_id = 1;
SELECT u.username, COUNT(p.id) AS post_count FROM users u LEFT JOIN posts p ON p.user_id = u.id GROUP BY u.id;
