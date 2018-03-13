BIN_DIR=../bin

CC=gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
# CFLAGS=-lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result
CFLAGS=-lm -pthread -O3 -Wall -march=native -funroll-loops  -Wno-unused-result -I/usr/include/malloc # OS X

all: word2vec word2phrase distance word-analogy compute-accuracy

word2vec : word2vec.c
	$(CC) word2vec.c -o ${BIN_DIR}/word2vec_csv $(CFLAGS)
word2phrase : ../c/word2phrase.c
	$(CC) ../c/word2phrase.c -o ${BIN_DIR}/word2phrase $(CFLAGS)
distance : ../c/distance.c
	$(CC) ../c/distance.c -o ${BIN_DIR}/w2v-distance $(CFLAGS)
word-analogy : ../c/word-analogy.c
	$(CC) ../c/word-analogy.c -o ${BIN_DIR}/w2v-word-analogy $(CFLAGS)
compute-accuracy : ../c/compute-accuracy.c
	$(CC) ../c/compute-accuracy.c -o ${BIN_DIR}/w2v-compute-accuracy $(CFLAGS)

clean:
	pushd ${BIN_DIR} && rm -rf word2vec_csv word2phrase distance word-analogy compute-accuracy; popd
