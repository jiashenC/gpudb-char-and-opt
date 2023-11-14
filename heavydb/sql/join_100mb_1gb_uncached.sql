drop table join_1;
drop table join_2;
create table join_1 (pk INT, attr INT);
create table join_2 (pk INT, attr INT);
copy join_1
from './data/storage/join_dim_26214400.parquet' with (parquet = 'true');
copy join_2
from './data/storage/join_fact_268435456.parquet' with (parquet = 'true');
select *
from join_1,
    join_2
where join_1.pk = join_2.pk;