GASNET_BACKTRACE=1  GASNET_MASTERIP='127.0.0.1' GASNET_SPAWN=-L SSH_SERVERS="localhost localhost localhost localhost localhost" amudprun -np 1 $@ -ll:gsize 64 -ll:csize 2048
