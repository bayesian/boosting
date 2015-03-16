FOLLY=$(HOME)/folly
THRIFT=$(HOME)/thrift-0.9.2/lib/cpp

all: src/*cpp include/*h
	g++ src/*cpp \
		-std=gnu++11 \
		-Iinclude -I$(FOLLY) -I$(THRIFT)/src \
		-o boosting_exec \
		-ldouble-conversion \
		-lglog \
		-lgflags \
		-L$(FOLLY)/folly/.libs \
		-lfolly \
		-L$(THRIFT)/.libs \
		-lthrift \
		-lthriftnb
