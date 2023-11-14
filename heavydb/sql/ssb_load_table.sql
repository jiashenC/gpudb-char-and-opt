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