//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <unordered_map>
#include <set>
#include <vector>
#include <string>
#include <random>
#include <iostream>

#define MAX_STRING_VOCAB 5000
#define MAX_STRING 100
#define MAX_STRING_FILE 1000
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000  // rows of more than 1k columns are unusual
#define MAX_CODE_LENGTH 40
#define MAX_TABLE_LENGTH 100000
#define MODDER 1

using namespace std;

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
  char *set;
};

char train_file[MAX_STRING_FILE], output_file[MAX_STRING_FILE];
char save_vocab_file[MAX_STRING_FILE], read_vocab_file[MAX_STRING_FILE];
long long id_files[MAX_TABLE_LENGTH];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 0, num_threads = 8, min_reduce = 0;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, iter_saved = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0;
int negative = 10;
int fnegative = 10;
int positive=0;

const int table_size = 1e8;
int *table;

// random machinery

unsigned cheap_rand(int seed){
    seed = ((seed * 7261) + 1) % 32768; 
    return seed;
}

void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

typedef struct {
  int index;
  int inQuotes;
  int Tindex;
  char * word;
  int currentFile;
  int currentThread;
  // Dictionary* set;
} CSVLoader;

void csv_new(CSVLoader *load,int thread)
{
  load->index = 0;
  load->inQuotes = 0;
  load->Tindex = 0;
  load->currentFile = 0;
  load->currentThread = 0;
}

unordered_map<int, unordered_map<int, float> > uniqueness_map; 
unordered_map<int, unordered_map<int, vector<string> > > sample_map; 
unordered_map<int, set<string> > data_map; 

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin, CSVLoader* load) {
  int a = 0, ch;
  load->Tindex = load->index;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if (ch == ',') {
      load->Tindex += 1;
    }
    if ((ch == ',') || (ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        // ASSIGNING 0 to carriage return
        strcpy(word, (char *)"</s>"); // that's because /s is 0
        load->Tindex = 0;
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, CSVLoader* load) {
  char word[MAX_STRING];
  ReadWord(word, fin,load);
  if(strcmp(word, "~R!RR*~") == 0){
    // printf("NEW FILE\n" );
    load->currentFile += 1;
    // continue;
    return -1;
  }
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word, char* index) {
  //TODO no index DONE
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  vocab[vocab_size].set = (char *)calloc(MAX_STRING_VOCAB, sizeof(char));
  vocab[vocab_size].set[0] = '\0';
  if (strstr(vocab[vocab_size].set, index) == NULL) {
    strcat(vocab[vocab_size].set,index);
  }
  strncpy(vocab[vocab_size].word, word,length);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  // min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}


void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;

  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>",(char*) "-1");
  CSVLoader load;
  csv_new(&load,0);
  int lines_per_file = 0;
  while (1) {
    ReadWord(word, fin,&load);

    // For FOCUSED negative sampling
    if(strcmp(word, "</s>")) { // if a normal word
      // store word in column_id - set
      data_map[load.index].insert(string(word)); 
      //printf("adding word to data_map: %s", word);
    }
    else { // if end of line
      // end of line
      lines_per_file++; // will use this later to compute uniqueness
    }

    if(!strcmp(word, "~R!RR*~")){
      load.index = 0;
      load.inQuotes = 0;
      id_files[load.currentFile] = ftell(fin); //WHEN IT ENDS
      // printf("%lli\n", id_files[load.currentFile]);

      // Done with reading this file
      // compute uniqueness and store data for sampling
      printf("finished reading file with lines: %d \n", lines_per_file);
      unordered_map<int, set<string> >::iterator it;
      for (it = data_map.begin(); it != data_map.end(); it++) {
        int col_id = it->first;
        set<string> data = it->second;
        printf("data size: %d \n", data.size());
        printf("total: %d \n", lines_per_file); 
        float uniqueness = (float)data.size() / (float)lines_per_file;
        printf("uniqueness of: %d - %d is: %.3f \n", load.currentFile, col_id, uniqueness);
        uniqueness_map[load.currentFile][col_id] = uniqueness;
        set<string>::iterator it2;
        for (it2 = data.begin(); it2 != data.end(); it2++) {
          sample_map[load.currentFile][col_id].push_back(*it2);
        }
      }

      //fflush(stdout);
      //sleep(20);
      // reset variables
      data_map.clear();
      lines_per_file = 0;

      load.currentFile += 1;
      continue;
    }
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    char snum[10];
    sprintf(snum, "%i_%i,", load.currentFile, load.index);
    // printf("%s %s\n",word,snum );
    if (i == -1) {
      a = AddWordToVocab(word,snum);
      vocab[a].cn = 1;
    } else {
      vocab[i].cn++;
      if (strstr(vocab[i].set, snum) == NULL) {
        strcat(vocab[i].set,snum);
      }
    }
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
    load.index = load.Tindex;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  //TODO DOES NOT WORK
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  CSVLoader load;
  csv_new(&load,0);
  while (1) {
    ReadWord(word, fin, &load);
    if(strcmp(word, "~R!RR*~") == 0){
      load.currentFile += 1;
      continue;
    }
    if (feof(fin)) break;
    //TODO
    a = AddWordToVocab(word,(char*) "TODO");
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  CreateBinaryTree();
}

void *TrainModelThread(void *id) {
  
  // random gen for this thread
  random_device rd;
  mt19937 eng(rd());
  uniform_int_distribution<> distr(0, 150000);
  int random_seed = distr(eng);


  long long a, b, d, cw, word, last_word, sentence_length, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  int pos[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  long long currentLocation = file_size / (long long)num_threads * (long long)id;
  int currentFile = 0;
  while(currentLocation > id_files[currentFile]) {
    currentFile += 1;
  }
  //printf("THREAD IS USING %i\n", currentFile);
  while (1) {
    //printf("word count: %d \n", word_count);
    //if (word_count - last_word_count > 10000) {
    if (word_count - last_word_count > 100) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }

    // read a row
    CSVLoader load;
    if (sentence_length == 0) {
      csv_new(&load,0);
      load.currentFile = currentFile;
      while (1) {
        word = ReadWordIndex(fi,&load);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;  // this means the row has been read
        sen[sentence_length] = word;
        pos[sentence_length] = load.index;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }

    // if file is done or processed more words than assigned to this thread finish iteration and prepare next
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }

    word = sen[sentence_position];
    //printf("pos word index: %d \n", word);
    //printf("pos word: %s \n", vocab[word].word);
    //sleep(3);
    //fflush(stdout);
    int indexI = pos[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    //NEW MODIFIED WINDOW
    window = sizeof(sen);
    //printf("window size before loop: %d  ", window);
    b = next_random % window;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;

      // ENTIRE ROW
      //fflush(stdout);
      //sleep(2);
      //printf("window size before loop: %d  ", window);
      //printf("sentence length: %d  ", sentence_length);
      for (a = 0; a < window * 2 + 1 - b; a++) if (a != window) {
        // Essentially we are adding all the words in entire column.

        //printf("a: %lld  ", a);

        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        //printf("c: %lld  ", c);
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        //printf("word to insert id: %d \n", sen[c]);
        //printf("word to insert: %s \n", vocab[sen[c]].word);
        cw++;
      }

      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        // NEGATIVE SAMPLING
        //fflush(stdout);
        //sleep(3);
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            //char * dict = vocab[target].set;
            char snum[10];
            sprintf(snum, "%i_%i,", load.currentFile ,indexI);

            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          //printf("target: %d \n", target);
          //vocab[vocab[target]].word
          //printf("target_word: %s \n", vocab[target].word);
          //printf("label: %d  ", label);
          l2 = target * layer1_size;
          f = 0;

          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha * MODDER;
          else if (f < -MAX_EXP) g = (label - 0) * alpha * MODDER;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha * MODDER;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];

          // focused relational neg sampling
          if (label == 1) {
            int file_id = load.currentFile;
            unordered_map<int, vector<string> > &file_data = sample_map[file_id];
            int this_column_id = load.index;
            unordered_map<int, vector<string> >::const_iterator it;
            for (it = file_data.begin(); it != file_data.end(); it++) {
              int that_column_id = it->first;
              float uniqueness = uniqueness_map[file_id][that_column_id];
              int total_samples = fnegative * uniqueness;
              if (total_samples == 0) {
                total_samples = 1;
              }
              //printf("uniqueness: %.2f - total_samples: %d \n", uniqueness, total_samples);
              int samples = 0;
              vector<string> &to_sample_from = file_data[that_column_id];
              int range = to_sample_from.size() - 1;
              if (range == 0) {
		continue; // we may want to check these cases in more detail; is this destroying cols of the embedding?
	      }
              //printf("file_id: %d \n", file_id);
              //printf("col_id: %d \n", that_column_id);
              //printf("range: %d \n", range);
              set<long long> seen_indexes;
              //random_device rd;
              //mt19937 eng(rd());
              //uniform_int_distribution<> distr(0, range);
              //lfsr = (long long)id;
              //long long random_index = (long long)id; 
              //int random_index = lfsr_rand() * (long long)id;
              int total_attempts = 0;
              while (samples < total_samples) {
                total_attempts = total_attempts + 1;
                if (total_attempts > total_samples * 100) {
			total_attempts = 0;
			break; // tried too many times, not making progress on focused neg sampling -> abort
		}
                //int random_index = distr(eng);

                int random_index = cheap_rand(random_seed);
                random_seed = random_index;
                random_index = random_index % range; // enforce bounds

                //random_index = random_index + lfsr_rand() * (long long)id;
                //OLD random_index = (random_index * (unsigned long long)2521917 + 11) % range;
                //random_index = lfsr_rand();//(random_index + (long long)id * 11);
                //if (random_index >= range) {
	//		random_index = random_index % range;
	//	}
                //if (random_index >= range || random_index < 0) {
                //    printf("will break");
               // }
               	//printf("idx: %lli\n", random_index);
               	//printf("seed: %lli\n", random_seed);
                //printf("ran: %lli\n", range);

                //// make sure it's without replacement
                //bool is_in = seen_indexes.find(random_index) != seen_indexes.end();
                //if (is_in) {
                //  continue;
               // }
                //else {
                //  seen_indexes.insert(random_index);
               // }
                string &s_word = to_sample_from[random_index];
                //char *sampled_word = new char[s_word.length() + 1];
                char sampled_word[s_word.length() + 1]; 
                strcpy(sampled_word, s_word.c_str());
    	 	// obtain here the position of the sampled word in the vocabulary
  		target = SearchVocab(sampled_word);
     		if (target == -1) {
                        //printf(sampled_word);
              		//printf("\n");
			//printf("TARGET IS -1\n");
  			continue;
		}
                label = 0; // these are all negative samples
                //target = 23;

                //printf("neg sample: %d \n", target);
                //printf("neg sample word: %s \n", vocab[target].word);

                //sampled_word = s_word.c_str();
                //printf("sampled_word: %s \n", sampled_word);
                //std::cout <<  "sampled-word: " << sampled_word;
                // make sure it's not the positive word

                // FIXME: check it does not collide with any word in the row!!
                set<long long> banned_words;
                for (int i = 0; i < sentence_length; i++) {
   	 	  long long word_index = sen[i];
                  banned_words.insert(word_index);
                  //printf("ban word to insert id: %d \n", word_index);
                  //printf("ban word to insert: %s \n", vocab[word_index].word);
                }

                bool is_in2 = banned_words.find(target) != banned_words.end();
                if (is_in2) {
                  continue; // sampled word in positive row tuple
                }
                // if nothing else, use word as negative sample
                l2 = target * layer1_size;
           	f = 0;

           	for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          	if (f > MAX_EXP) g = (label - 1) * alpha * MODDER;
          	else if (f < -MAX_EXP) g = (label - 0) * alpha * MODDER;
          	else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha * MODDER;
          	for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          	for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];

                samples++; // valid sample, go on
              }

            }

          } // END FOCUSED negative sampling

        }
        // hidden -> in
        for (a = 0; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}


/*
void do_neg_train(long long layer1_size, long long l2, int label, int f, float g){
	for (int c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
        if (f > MAX_EXP) g = (label - 1) * alpha * MODDER;
        else if (f < -MAX_EXP) g = (label - 0) * alpha * MODDER;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha * MODDER;
        for (int c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
}
*/

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  int actual_iter = iter;
  int total_iter = 0;
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  char buffer[MAX_STRING];
  sprintf(buffer, "_v%lld_n%d_i%i_dynamic_window_only.txt", layer1_size,negative,actual_iter);
  char output_filename[MAX_STRING_FILE];
  strcpy(output_filename,output_file);
  // strcat(output_filename, buffer);
  fo = fopen(output_filename, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\t-iter-saved <int>\n");
    printf("\t\t0 for only saving iters vector once, else 1 is every power of 2... class defaults to 0\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  //if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-fnegative", argc, argv)) > 0) fnegative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-positive", argc, argv)) > 0) positive = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter-saved", argc, argv)) > 0) iter_saved = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }

  TrainModel();
  return 0;
}




