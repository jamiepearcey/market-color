-- eventgraph :: DUCKLAKE attach.
-- Two catalog options. The coherent single-database story uses Postgres as BOTH
-- the DuckLake catalog AND the reference/lifecycle store (schema/*.sql):
--
--   INSTALL ducklake; LOAD ducklake;
--   INSTALL postgres; LOAD postgres;
--   ATTACH 'ducklake:postgres:dbname=eventgraph host=localhost' AS eg
--          (DATA_PATH 'eg_data/');
--
-- Or a zero-config local catalog (DuckDB file) for dev:
--
INSTALL ducklake; LOAD ducklake;
ATTACH 'ducklake:eventgraph_catalog.ducklake' AS eg (DATA_PATH 'eg_data/');

USE eg;
