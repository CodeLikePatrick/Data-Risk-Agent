-- Data profiling checks, Hive flavour.
-- These are the warehouse equivalents of the in-memory quality checks in
-- ecomrisk/data.py and the feature pipeline.

-- Completeness: null rate on key fields.
SELECT
    'merchants'                                  AS tbl,
    COUNT(*)                                     AS rows,
    SUM(CASE WHEN merchant_id IS NULL THEN 1 ELSE 0 END) AS null_merchant_id,
    SUM(CASE WHEN category    IS NULL THEN 1 ELSE 0 END) AS null_category,
    SUM(CASE WHEN address_raw IS NULL THEN 1 ELSE 0 END) AS null_address
FROM merchants;

-- Uniqueness: one row per merchant.
SELECT merchant_id, COUNT(*) AS n
FROM merchants
GROUP BY merchant_id
HAVING COUNT(*) > 1;

-- Validity: prices must be positive and inside a sane range.
SELECT
    COUNT(*) AS rows,
    SUM(CASE WHEN price <= 0 THEN 1 ELSE 0 END)  AS non_positive_price,
    SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) AS null_price
FROM products;

-- Distribution: price summary by category.
SELECT
    category,
    COUNT(*)       AS n,
    MIN(price)     AS price_min,
    percentile_approx(price, 0.5) AS price_median,
    MAX(price)     AS price_max,
    AVG(price)     AS price_avg,
    STDDEV(price)  AS price_std
FROM products
GROUP BY category;
