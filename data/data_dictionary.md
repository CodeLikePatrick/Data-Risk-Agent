# Data dictionary

## merchants

| Column | Type | Meaning |
| --- | --- | --- |
| merchant_id | string | Primary key, one row per merchant |
| name | string | Merchant display name |
| category | string | Product category the merchant belongs to |
| city | string | Claimed city |
| district | string | Claimed district |
| address_raw | string | Raw registered address text |
| lat, lon | float | Claimed coordinates |
| address_fake | boolean | Whether the registered address is fictitious (upstream parser flag) |
| device_group | integer or null | Shared device fingerprint group; null when unique |
| true_types | string | Pipe-separated injected violation types (label) |
| true_risk | integer | Ground-truth risk label, 1 risky / 0 normal |

## products

| Column | Type | Meaning |
| --- | --- | --- |
| product_id | string | Primary key |
| merchant_id | string | Foreign key to merchants |
| category | string | Product category |
| title | string | Product title text |
| price | float | Listed price |

## Grain and joins

`merchants` is one row per merchant. `products` is one row per listed product and
joins to `merchants` on `merchant_id`. Risk is scored at the merchant grain; the
product-level signals (price and title) are aggregated up to the merchant.
