
DROP TABLE IF exists tot_intab;
DROP TABLE IF exists filtered_cleaned_table;
DROP TABLE IF exists num_pids;
DROP TABLE IF exists wmv;
DROP TABLE IF exists dma_name;

--- number intab per DMA per day (P2-99)

CREATE TEMP TABLE tot_intab AS (
  select date, dma_name, count(distinct(pid)) as total_intab
    from(
         (select A.date, A.pid, B.household_number 
            from dev.nielsen_in_tab A
          INNER JOIN dev.nielsen_market_breaks B 
          ON A.pid = B.pid) AS C
  INNER JOIN dev.l5m_dmas 
  ON dev.l5m_dmas.hhid =c.household_number)
  group by date, dma_name
  order by date, dma_name
  );
  
-- get a filtered, cleaned table of all viewing events
CREATE TEMP TABLE filtered_cleaned_table AS (
select t1.*, t2.cleaned_program
      from (
        select dev.l5m_all_minute.*, dev.l5m_dmas.dma_name, date_part( 'dow', date ) as dow 
          from dev.l5m_all_minute 
        INNER JOIN dev.l5m_dmas 
        ON dev.l5m_all_minute.hhid=dev.l5m_dmas.hhid
        WHERE telecast_start >= '2017-01-01' AND telecast_start < '2018-01-01' 
        AND time_shifted_viewing='live' AND dow >=2 AND dow <=6) t1
    INNER JOIN 
      (select program, cleaned_program, network_code 
        from dev.nim_program_match
      WHERE cleaned_program in PROGS_TO_REPLACE) t2
    ON t1.program = t2.program AND t1.network_code=t2.network_code
    );

--- get number of distinct pids seen for that telecast
CREATE TEMP TABLE num_pids AS (
select telecast_start, dma_name, COUNT(DISTINCT(pid))
from filtered_cleaned_table
group by cleaned_program, telecast_start, dma_name 
order by dma_name, telecast_start);

--- get WMV (each row is a viewing event for a pid for a particular telecast)
CREATE TEMP TABLE wmv AS (
select dma_name, pid, telecast_start, cleaned_program, sum(weighted_minutes_viewed) as wmv 
from filtered_cleaned_table
group by pid, cleaned_program, telecast_start, dma_name 
order by dma_name, pid, cleaned_program);

---get number of zero minute viewing events for each dma across all telecasts
select dma_name, sum(num_zeros)
  FROM (
    SELECT t1.dma_name, t1.date, t2.total_intab - t1.count as num_zeros
      FROM(
        select dma_name, count, telecast_start::date as date
          from num_pids) t1
    INNER JOIN tot_intab t2
    ON t1.date=t2.date AND t1.dma_name = t2.dma_name)
GROUP BY dma_name
ORDER BY dma_name;


