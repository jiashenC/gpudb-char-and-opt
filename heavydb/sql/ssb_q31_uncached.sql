drop table customer;
drop table ddate;
drop table lineorder;
drop table part;
drop table supplier;
create table lineorder (
    lo_orderkey integer not null,
    lo_linenumber integer not null,
    lo_custkey integer not null,
    lo_partkey integer not null,
    lo_suppkey integer not null,
    lo_orderdate integer not null,
    lo_orderpriority text not null,
    lo_shippriority text not null,
    lo_quantity integer not null,
    lo_extendedprice integer not null,
    lo_ordtotalprice integer not null,
    lo_discount integer not null,
    lo_revenue integer not null,
    lo_supplycost integer not null,
    lo_tax integer not null,
    lo_commitdate integer not null,
    lo_shopmode text not null,
    dummy double
);
create table part (
    p_partkey integer not null,
    p_name text not null,
    p_mfgr text not null,
    p_category text not null,
    p_brand1 text not null,
    p_color text not null,
    p_type text not null,
    p_size integer not null,
    p_container text not null,
    dummy double
);
create table supplier (
    s_suppkey integer not null,
    s_name text not null,
    s_address text not null,
    s_city text not null,
    s_nation text not null,
    s_region text not null,
    s_phone text not null,
    dummy double
);
create table customer (
    c_custkey integer not null,
    c_name text not null,
    c_address text not null,
    c_city text not null,
    c_nation text not null,
    c_region text not null,
    c_phone text not null,
    c_mktsegment text not null,
    dummy double
);
create table ddate (
    d_datekey integer not null,
    d_date text not null,
    d_dayofweek text not null,
    d_month text not null,
    d_year integer not null,
    d_yearmonthnum integer not null,
    d_yearmonth text not null,
    d_daynuminweek integer not null,
    d_daynuminmonth integer not null,
    d_daynuminyear integer not null,
    d_monthnuminyear integer not null,
    d_weeknuminyear integer not null,
    d_sellingseasin text not null,
    d_lastdayinweekfl integer not null,
    d_lastdayinmonthfl integer not null,
    d_holidayfl integer not null,
    d_weekdayfl integer not null,
    dummy double
);
copy customer
from './data/storage/sf[sfph]_customer.parquet' with (parquet = 'true');
copy ddate
from './data/storage/sf[sfph]_date.parquet' with (parquet = 'true');
copy lineorder
from './data/storage/sf[sfph]_lineorder.parquet' with (parquet = 'true');
copy part
from './data/storage/sf[sfph]_part.parquet' with (parquet = 'true');
copy supplier
from './data/storage/sf[sfph]_supplier.parquet' with (parquet = 'true');
select c_nation,
    s_nation,
    d_year,
    sum(lo_revenue) as revenue
from customer,
    lineorder,
    supplier,
    ddate
where lo_custkey = c_custkey
    and lo_suppkey = s_suppkey
    and lo_orderdate = d_datekey
    and c_region = 'ASIA'
    and s_region = 'ASIA'
    and d_year >= 1992
    and d_year <= 1997
group by c_nation,
    s_nation,
    d_year
order by d_year asc,
    revenue desc;