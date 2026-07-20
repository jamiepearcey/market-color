-- eventgraph :: DUCKLAKE formal-plane options layer (append-only).
-- Two granularities, populated by scripts/options_surface.py:
--   option_iv      NAME-LEVEL daily IV summary (HF gauss314/options-IV-SP500,
--                  Apache-2.0: moneyness-ladder IVs + HV terms + VIX; 3,893
--                  symbols, 2019-10-14..2023-07-28). The free single-name
--                  implied layer: vol risk premium = atm_iv - hv_20, wing
--                  skew = dotm_iv - atm_iv, all at query time.
--   option_surface UNDERLIER daily surface summary derived from full EOD
--                  chains (Kaggle SPY 2010-2023 etc.): front/next ATM IV,
--                  put skew, term slope, straddle-implied move. Joins to
--                  calendar_event on date for implied-move-into-event.

CREATE TABLE IF NOT EXISTS eg.option_iv (
  symbol      VARCHAR,
  date        DATE,
  ditm_iv     DOUBLE, itm_iv  DOUBLE, sitm_iv DOUBLE, atm_iv DOUBLE,
  sotm_iv     DOUBLE, otm_iv  DOUBLE, dotm_iv DOUBLE,
  calls_traded BIGINT, puts_traded BIGINT,
  calls_oi     BIGINT, puts_oi     BIGINT,
  contracts    BIGINT, expirations INTEGER, strikes_spread DOUBLE,
  hv_20 DOUBLE, hv_40 DOUBLE, hv_60 DOUBLE, hv_90 DOUBLE, hv_180 DOUBLE,
  vix   DOUBLE,
  source VARCHAR                    -- 'hf:gauss314/options-IV-SP500'
);

CREATE TABLE IF NOT EXISTS eg.option_surface (
  underlier     VARCHAR,
  date          DATE,
  spot          DOUBLE,
  front_expiry  DATE,
  front_dte     INTEGER,
  atm_iv_front  DOUBLE,
  atm_iv_next   DOUBLE,            -- next monthly-ish expiry -> term slope
  term_slope    DOUBLE,            -- atm_iv_next - atm_iv_front
  put_skew      DOUBLE,            -- ~25d-put IV minus ATM IV (crash fear)
  implied_move  DOUBLE,            -- ATM straddle / spot to front expiry
  n_contracts   BIGINT,
  source        VARCHAR
);

-- Implied move standing before each formal calendar event (macro fear gauge):
-- the most recent surface row strictly BEFORE the event, within 5 days.
CREATE OR REPLACE VIEW eg.v_event_implied AS
SELECT c.series_id, c.event_time, s.underlier, s.date AS surface_date,
       s.atm_iv_front, s.put_skew, s.term_slope, s.implied_move
FROM eg.calendar_event c
JOIN eg.option_surface s
  ON s.date < CAST(c.event_time AS DATE)
 AND s.date >= CAST(c.event_time AS DATE) - INTERVAL 5 DAY
QUALIFY row_number() OVER (PARTITION BY c.series_id, c.event_time, s.underlier
                           ORDER BY s.date DESC) = 1;
