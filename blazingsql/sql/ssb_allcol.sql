select lo_orderdate
from lineorder
where lo_discount between 1 and 3
    and lo_quantity < 10000;