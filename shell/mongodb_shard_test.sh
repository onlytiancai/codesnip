set -e
set -x

# Sanity Checks
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Must be run with root privileges."
    exit 1
fi

# config
MONGOD=/usr/local/mongodb/bin/mongod
MONGOS=/usr/local/mongodb/bin/mongos
MONGODB_USER=mongodb
MONGODB_DATA_DIR=/data/db
MONGODB_LOG_DIR=/data/log

# create the user if non-existent
if ! id $MONGODB_USER >/dev/null ; then
    adduser --system $MONGODB_USER
fi

# create path
function mkdir_path(){
    if [ ! -d $1 ]; then
        mkdir -p $1
        chown $MONGODB_USER $1
    fi
}

# start mongod 
function start_mongod(){
    if ! nc -vz localhost $2; then
        sudo -u $MONGODB_USER $MONGOD --fork --dbpath=$MONGODB_DATA_DIR/$1 --port $2 --logpath $MONGODB_DATA_DIR/mongodb.$1.log
        while ! nc -vz localhost $2; do
            sleep 1
        done
    fi
}

# create data path
for dir in $MONGODB_DATA_DIR/configdb $MONGODB_DATA_DIR/shard1 $MONGODB_DATA_DIR/shard2; do
    mkdir_path $dir    
done

# config server start
start_mongod configdb 2222


# router server start
if ! nc -vz localhost 3333; then
    sudo -u $MONGODB_USER $MONGOS --fork --port 3333 --configdb=localhost:2222 --logpath $MONGODB_LOG_DIR/mongods.log
    while ! nc -vz localhost 3333; do
        sleep 1
    done
fi

# mongodb shard
start_mongod shard1 4444 
start_mongod shard2 5555 
