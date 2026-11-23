CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY CHECK (customer_id > 0),
    customer_name TEXT NOT NULL,
    city TEXT NOT NULL
) STRICT;

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY CHECK (product_id > 0),
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    list_price_clp INTEGER NOT NULL CHECK (list_price_clp >= 0)
) STRICT;

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY CHECK (order_id > 0),
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date TEXT NOT NULL CHECK (length(order_date) = 10),
    status TEXT NOT NULL CHECK (status IN ('completed', 'pending', 'canceled'))
) STRICT;

CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price_clp INTEGER NOT NULL CHECK (unit_price_clp >= 0),
    PRIMARY KEY (order_id, product_id)
) STRICT;
