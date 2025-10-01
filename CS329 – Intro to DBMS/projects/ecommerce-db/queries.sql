-- ecommerce-db/queries.sql
SELECT c.name, o.id AS order_id, SUM(p.price * oi.qty) AS total
FROM customers c
JOIN orders o ON o.customer_id = c.id
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON p.id = oi.product_id
GROUP BY o.id;
