drop table aggregate_1;
create table aggregate_1 (pk INT, attr INT);
copy aggregate_1
from './data/storage/aggregate_268435456_10000.parquet' with (parquet = 'true');
select sum(aggregate_1.attr)
from aggregate_1
group by aggregate_1.pk;