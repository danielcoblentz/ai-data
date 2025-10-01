# Normalization examples

Example: customer orders where order items were stored as comma-separated values (bad). Show how to split into orders and order_items tables to achieve 1NF/3NF.

-- Before: orders(order_id, customer_id, item_list)
-- After: orders(order_id, customer_id), order_items(order_item_id, order_id, product_id, qty)
