-- Feature engineering, Hive flavour.
-- Mirrors ecomrisk/engine.py::build_features so the same logic can run in a
-- warehouse. `merchants.address_fake` is an upstream flag produced by an
-- address parser; the remaining signals are computed here.

WITH category_price AS (
    SELECT
        category,
        percentile_approx(price, 0.5) AS price_median,
        COALESCE(STDDEV(price), 1)    AS price_std
    FROM products
    GROUP BY category
),
product_stats AS (
    SELECT
        p.merchant_id,
        MAX(ABS((p.price - c.price_median) / c.price_std))       AS price_zmax,
        AVG(CASE WHEN p.price < 0.1 * c.price_median THEN 1 ELSE 0 END) AS price_low_ratio,
        AVG(CASE WHEN SIZE(COLLECT_SET(SPLIT(p.title, ''))) / LENGTH(p.title) < 0.45
                 THEN 1 ELSE 0 END)                               AS spam_ratio
    FROM products p
    JOIN category_price c ON p.category = c.category
    GROUP BY p.merchant_id
),
group_flags AS (
    SELECT
        merchant_id,
        CASE WHEN COUNT(*) OVER (PARTITION BY device_group) > 1 THEN 1 ELSE 0 END AS group_flag
    FROM merchants
    WHERE device_group IS NOT NULL
)
SELECT
    m.merchant_id,
    m.category,
    CASE WHEN m.address_fake THEN 1 ELSE 0 END                 AS addr_fake,
    -- Coordinates outside the claimed city bounds would be flagged here by a
    -- user-defined function over the district boundary table.
    0                                                           AS addr_contradiction,
    COALESCE(s.price_zmax, 0)                                   AS price_zmax,
    COALESCE(s.price_low_ratio, 0)                              AS price_low_ratio,
    COALESCE(s.spam_ratio, 0)                                   AS spam_ratio,
    COALESCE(g.group_flag, 0)                                   AS group_flag
FROM merchants m
LEFT JOIN product_stats s ON s.merchant_id = m.merchant_id
LEFT JOIN group_flags  g ON g.merchant_id = m.merchant_id;
