-- Issuers present in events but missing from outcomes (survivorship-style coverage check).
SELECT DISTINCT LOWER(e.issuer_id) AS issuer_id FROM events e WHERE LOWER(e.issuer_id) NOT IN (SELECT LOWER(issuer_id) FROM outcomes WHERE issuer_id IS NOT NULL) ORDER BY 1
