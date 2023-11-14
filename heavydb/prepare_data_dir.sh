# create data directory
mkdir heavydb/heavydb_src/build/data-part3
mkdir heavydb/heavydb_src/build/data-part4
mkdir heavydb/heavydb_src/build/data-part5
mkdir heavydb/heavydb_src/build/data-part6

# init data directory
./heavydb/heavydb_src/build/bin/initheavy heavydb/heavydb_src/build/data-part3
./heavydb/heavydb_src/build/bin/initheavy heavydb/heavydb_src/build/data-part4
./heavydb/heavydb_src/build/bin/initheavy heavydb/heavydb_src/build/data-part5
./heavydb/heavydb_src/build/bin/initheavy heavydb/heavydb_src/build/data-part6

# copy config files to heavydb directory
cp heavydb_part3.conf ./heavydb
cp heavydb_part4.conf ./heavydb
cp heavydb_part5.conf ./heavydb
cp heavydb_part6.conf ./heavydb