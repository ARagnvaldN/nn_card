CC_FLAGS=--std=c++11 -w

all:
	@g++ ${CC_FLAGS} lenet.cpp -o run && ./run input7.bin lenet.bin \
	&& g++ ${CC_FLAGS} card_lenet.cpp -o card \
	&& ./card input/input0.bin lenet.bin; echo $$? \
	&& ./card input/input1.bin lenet.bin; echo $$? \
	&& ./card input/input2.bin lenet.bin; echo $$? \
	&& ./card input/input3.bin lenet.bin; echo $$? \
	&& ./card input/input4.bin lenet.bin; echo $$? \
	&& ./card input/input5.bin lenet.bin; echo $$? \
	&& ./card input/input6.bin lenet.bin; echo $$? \
	&& ./card input/input7.bin lenet.bin; echo $$? \
	&& ./card input/input8.bin lenet.bin; echo $$? \
	&& ./card input/input9.bin lenet.bin; echo $$?
