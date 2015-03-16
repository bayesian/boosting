#!/bin/bash

FOLLY=$HOME/folly
THRIFT=$HOME/thrift-0.9.2/lib/cpp

LIB_PATH=${FOLLY}/folly/.libs:${THRIFT}/.libs:$LD_LIBRARY_PATH

EXEC_PATH=$(dirname $0)/boosting_exec
echo $@
LD_LIBRARY_PATH=${LIB_PATH} $HOME/boosting/boosting_exec $@
