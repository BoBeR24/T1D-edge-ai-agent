---
name: dawn-phenomenon
description: Questions about a dawn phenomenon, i.e. an early-morning glucose rise (around 3 AM vs around 6 AM).
tools: [get_cgm_analytics_from_db_range]
---
Goal: compare the average glucose around 3 AM with the average glucose around 6 AM over the requested period.

Steps:
1. Call get_cgm_analytics_from_db_range ONCE for the whole requested period, with hourly=true.
   - start_iso = first day at 00:00:00+00:00, end_iso = last day at 23:59:59+00:00.
   - Example (illustrative dates): start_iso="2030-03-01T00:00:00+00:00", end_iso="2030-03-10T23:59:59+00:00", hourly=true.
   - Do not call any other data tool: the hourly averages contain everything you need.
2. In the "[hourly avg]" line, read the value for 03h (pre-dawn) and for 06h (wake).
3. Decide:
   - 06h clearly higher than 03h (a rise of about 1 mmol/L or more): a dawn phenomenon is likely.
   - Otherwise (similar values, or 06h lower): no clear dawn phenomenon.
4. Call log_conclusion with findings as a JSON object of numbers, for example:
   findings='{"avg_pre_dawn": 7.2, "avg_wake": 7.9}'  (illustrative values)
5. Then reply to the user in 2-4 plain sentences: both averages in mmol/L, the difference,
   your conclusion, and a short note that this is not medical advice.
   Never leave the final reply empty.