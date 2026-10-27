-- Отмены считаются по всем заказам; своевременность — только по доставленным с валидными датами.
SELECT STRFTIME(order_purchase_timestamp,'%Y-%m') AS month,
       COUNT(*) AS orders,
       COUNT(*) FILTER (WHERE order_status='canceled') AS canceled,
       COUNT(*) FILTER (WHERE timing_valid) AS delivered,
       COUNT(*) FILTER (WHERE timing_valid AND delay_days>0) AS late,
       AVG(CASE WHEN timing_valid THEN (delay_days>0)::INTEGER END) AS late_rate,
       AVG(CASE WHEN timing_valid THEN delivery_days END) AS delivery_days_mean
FROM analysis_mart GROUP BY 1 ORDER BY 1;
