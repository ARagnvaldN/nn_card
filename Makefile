CC_FLAGS=--std=c++11 -w

all:
	@g++ ${CC_FLAGS} lenet.cpp -o run && ./run input7.bin lenet.bin \
	&& g++ ${CC_FLAGS} card_lenet.cpp -o card \
	&& ./card input0.bin lenet.bin; echo $$? \
	&& ./card input1.bin lenet.bin; echo $$? \
	&& ./card input2.bin lenet.bin; echo $$? \
	&& ./card input3.bin lenet.bin; echo $$? \
	&& ./card input4.bin lenet.bin; echo $$? \
	&& ./card input5.bin lenet.bin; echo $$? \
	&& ./card input6.bin lenet.bin; echo $$? \
	&& ./card input7.bin lenet.bin; echo $$? \
	&& ./card input8.bin lenet.bin; echo $$? \
	&& ./card input9.bin lenet.bin; echo $$?
