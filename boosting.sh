#!/bin/bash

FOLLY=$HOME/folly
THRIFT=$HOME/thrift-0.9.2/lib/cpp

LIB_PATH=${FOLLY}/folly/.libs:${THRIFT}/.libs:$LD_LIBRARY_PATH

echo ${LIB_PATH}
LD_LIBRARY_PATH=${LIB_PATH} ./boosting_exec $@
