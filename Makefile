CC_FLAGS=--std=c++11 -w

.PHONY: all clean test lenet_card lenet

all: lenet lenet_card test

lenet:
	@g++ ${CC_FLAGS} lenet.cpp -o run

lenet_card:
	@g++ ${CC_FLAGS} card_lenet.cpp -o card

test:
	@./run input/input7.bin lenet.bin \
	./card input/input0.bin lenet.bin; echo -n $$? \
	&& ./card input/input1.bin lenet.bin; echo -n $$? \
	&& ./card input/input2.bin lenet.bin; echo -n $$? \
	&& ./card input/input3.bin lenet.bin; echo -n $$? \
	&& ./card input/input4.bin lenet.bin; echo -n $$? \
	&& ./card input/input5.bin lenet.bin; echo -n $$? \
	&& ./card input/input6.bin lenet.bin; echo -n $$? \
	&& ./card input/input7.bin lenet.bin; echo -n $$? \
	&& ./card input/input8.bin lenet.bin; echo -n $$? \
	&& ./card input/input9.bin lenet.bin; echo $$?

clean:
	@rm -rf run && rm -rf card
