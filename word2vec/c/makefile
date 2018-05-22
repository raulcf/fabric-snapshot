BIN_DIR=../bin

CC=g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
# CFLAGS=-lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result
CFLAGS=-lm -pthread -O3 -Wall -march=native -funroll-loops  -Wno-unused-result -I/usr/include/malloc -std=c++11 # OS X

all: word2vec word2vec_dynamic_window word2vec_dynamic_window_alternating word2vec_dynamic_window_only word2phrase distance word-analogy compute-accuracy

word2vec : word2vec.c
	$(CC) word2vec.c -o ${BIN_DIR}/word2vec $(CFLAGS)
db2vec : db2vec.c
	$(CC) db2vec.c -o ${BIN_DIR}/db2vec $(CFLAGS)
word2vec_dynamic_window : word2vec_dynamic_window.c
	$(CC) word2vec_dynamic_window.c -o ${BIN_DIR}/word2vec_dynamic_window $(CFLAGS)
word2vec_dynamic_window_only : word2vec_dynamic_window_only.c
	$(CC) word2vec_dynamic_window_only.c -o ${BIN_DIR}/word2vec_dynamic_window_only $(CFLAGS)
word2vec_dynamic_window_alternating : word2vec_dynamic_window_alternating.c
	$(CC) word2vec_dynamic_window_alternating.c -o ${BIN_DIR}/word2vec_dynamic_window_alternating $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o ${BIN_DIR}/word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o ${BIN_DIR}/w2v-distance $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o ${BIN_DIR}/w2v-word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o ${BIN_DIR}/w2v-compute-accuracy $(CFLAGS)

clean:
	pushd ${BIN_DIR} && rm -rf word2vec word2phrase distance word-analogy compute-accuracy; popd
