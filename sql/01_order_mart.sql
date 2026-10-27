-- Каждая таблица «один ко многим» сначала агрегируется до одного заказа.
-- Это защищает число заказов и денежные суммы от размножения при JOIN.
CREATE OR REPLACE TABLE order_mart AS
WITH item_totals AS (
    SELECT i.order_id, COUNT(*) AS items_count, COUNT(DISTINCT i.seller_id) AS sellers_count,
           COUNT(DISTINCT p.product_category_name) AS categories_count,
           SUM(i.price) AS items_value, SUM(i.freight_value) AS freight_value,
           SUM(p.product_weight_g)/1000.0 AS weight_kg,
           SUM(p.product_length_cm*p.product_height_cm*p.product_width_cm)/1000.0 AS volume_litres
    FROM raw_items i LEFT JOIN raw_products p USING (product_id)
    GROUP BY i.order_id
), seller_totals AS (
    SELECT order_id, seller_id, SUM(price) AS seller_value
    FROM raw_items GROUP BY order_id, seller_id
), main_seller AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY seller_value DESC, seller_id) AS rn
    FROM seller_totals
), category_totals AS (
    SELECT i.order_id, COALESCE(t.product_category_name_english,'unknown') AS category,
           SUM(i.price) AS category_value
    FROM raw_items i LEFT JOIN raw_products p USING(product_id)
    LEFT JOIN raw_translation t USING(product_category_name)
    GROUP BY i.order_id, COALESCE(t.product_category_name_english,'unknown')
), main_category AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY category_value DESC,category) AS rn
    FROM category_totals
), payments AS (
    SELECT order_id, SUM(payment_value) AS paid_value, COUNT(*) AS payments_count
    FROM raw_payments GROUP BY order_id
), reviews AS (
    SELECT *, COUNT(*) OVER(PARTITION BY order_id) AS review_records,
           ROW_NUMBER() OVER(PARTITION BY order_id ORDER BY review_answer_timestamp DESC NULLS LAST,
           review_creation_date DESC NULLS LAST, review_id DESC, review_score DESC) AS rn
    FROM raw_reviews
)
SELECT o.*, c.customer_state, c.customer_city,
       i.items_count,i.sellers_count,i.categories_count,i.items_value,i.freight_value,
       i.weight_kg,i.volume_litres,ms.seller_id AS main_seller_id,
       s.seller_state,mc.category,p.paid_value,p.payments_count,
       r.review_score,r.review_answer_timestamp,r.review_records,
       DATE_DIFF('day',CAST(o.order_purchase_timestamp AS DATE),
                 CAST(o.order_estimated_delivery_date AS DATE)) AS promise_days,
       EXTRACT(hour FROM o.order_purchase_timestamp) AS purchase_hour,
       EXTRACT(dow FROM o.order_purchase_timestamp) AS purchase_weekday,
       EXTRACT(month FROM o.order_purchase_timestamp) AS purchase_month,
       CASE WHEN c.customer_state=s.seller_state THEN 1 ELSE 0 END AS same_state,
       DATE_DIFF('day',CAST(o.order_estimated_delivery_date AS DATE),
                 CAST(o.order_delivered_customer_date AS DATE)) AS delay_days,
       (EPOCH(o.order_delivered_customer_date)-EPOCH(o.order_purchase_timestamp))/86400.0 AS delivery_days
FROM raw_orders o
LEFT JOIN raw_customers c USING(customer_id)
LEFT JOIN item_totals i USING(order_id)
LEFT JOIN main_seller ms ON o.order_id=ms.order_id AND ms.rn=1
LEFT JOIN raw_sellers s ON ms.seller_id=s.seller_id
LEFT JOIN main_category mc ON o.order_id=mc.order_id AND mc.rn=1
LEFT JOIN payments p ON o.order_id=p.order_id
LEFT JOIN reviews r ON o.order_id=r.order_id AND r.rn=1;
