drop table sample_1;
drop table sample_2;
create table sample_1 (pk INT, attr INT);
create table sample_2 (pk INT, attr INT);
copy sample_1
from './data/storage/sample_1.parquet' with (parquet = 'true');
copy sample_2
from './data/storage/sample_2.parquet' with (parquet = 'true');
select *
from sample_1,
    sample_2
where sample_1.pk = sample_2.pk;