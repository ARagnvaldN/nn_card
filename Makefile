CC_FLAGS=--std=c++11 -w

all:
	g++ ${CC_FLAGS} lenet.cpp -o run && ./run input7.bin lenet.bin \
	&& g++ ${CC_FLAGS} card_lenet.cpp -o card && ./card input7.bin lenet.bin; echo $$? 
