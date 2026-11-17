-- Count distinct issuers with at least one `default` exit, grouped by sector.
-- Run against tables loaded in research.duckdb from the discovery pack.
SELECT i.sector, COUNT(DISTINCT i.issuer_id) AS n_default_issuers FROM exits x JOIN issuers i ON LOWER(x.issuer_id) = LOWER(i.issuer_id) WHERE x.exit_reason = 'default' GROUP BY i.sector ORDER BY i.sector
