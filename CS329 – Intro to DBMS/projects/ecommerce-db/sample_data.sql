-- ecommerce-db/sample_data.sql
INSERT INTO customers (name, email) VALUES ('Bob', 'bob@shop.com');
INSERT INTO products (name, price) VALUES ('Widget', 9.99);
INSERT INTO orders (customer_id) VALUES (1);
INSERT INTO order_items (order_id, product_id, qty) VALUES (1, 1, 2);
