-- Share of events where published_at is more than 24 hours after event_time (UTC-normalized).
SELECT AVG(CASE WHEN CAST(REPLACE(published_at, 'Z', '+00') AS TIMESTAMPTZ) > (CAST(event_time AS TIMESTAMP) AT TIME ZONE event_tz) + INTERVAL 24 HOUR THEN 1.0 ELSE 0.0 END) AS share_late FROM (SELECT DISTINCT ON (event_id) * FROM events ORDER BY event_id) AS e
