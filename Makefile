CC_FLAGS=--std=c++11

all:
	g++ ${CC_FLAGS} lenet.cpp -o run && ./run input7.bin lenet.bin
