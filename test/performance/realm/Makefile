TESTDIRS = \
	event_latency \
	event_throughput \
	lock_chains \
	lock_contention \
	reducetest

all : run_all

run_all : $(TESTDIRS:%=run.%)
build_all : $(TESTDIRS:%=build.%)
clean_all : $(TESTDIRS:%=clean.%)

# since we're moving into subdirectories, LG_RT_DIR must be an absolute path
ABS_RT_DIR=$(shell cd $(LG_RT_DIR); pwd)

.NOTPARALLEL :

build.% :
	$(MAKE) -C $* LG_RT_DIR=$(ABS_RT_DIR) all

clean.% :
	$(MAKE) -C $* LG_RT_DIR=$(ABS_RT_DIR) clean

run.% :
	$(MAKE) -C $* LG_RT_DIR=$(ABS_RT_DIR) run
