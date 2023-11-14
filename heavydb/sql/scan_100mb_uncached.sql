drop table scan_1;
create table scan_1 (pk INT, attr INT);
copy scan_1
from './data/storage/join_dim_26214400.parquet' with (parquet = 'true');
select *
from scan_1;