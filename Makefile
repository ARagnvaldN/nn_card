CC_FLAGS=--std=c++11

all:
	g++ ${CC_FLAGS} test.cpp -o run && ./run
