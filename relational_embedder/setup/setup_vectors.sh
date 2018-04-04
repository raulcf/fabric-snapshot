#!/bin/bash

run_word2vec() {
  local vectors="$1"
  local negatives="$2"
  local iterations="$3"
  local type="$4"
  local delimiter=","
  if [ -n "$vectors" ]; then
    local l_vector
    while read -d "$delimiter" l_vector; do
      if [ -n "$negatives" ]; then
        local l_neg
        while read -d "$delimiter" l_neg; do
          if [ -n "$iterations" ]; then
            local l_iter
            while read -d "$delimiter" l_iter; do
              if [ "$type" == "3" ]; then
                $w2v_runtime_folder/word2vec -train $w2v_parsed_folder/$w2v_databasename.txt -output $w2v_vector_folder/"$w2v_databasename"_v"$l_vector"_n"$l_neg"_i"$l_iter".txt -size $l_vector -sample 1e-3 -negative $l_neg -hs 0 -binary 0 -cbow 1 -iter $l_iter -threads $w2v_threads
                # echo "==-=="
                $w2v_runtime_folder/word2vec -train $w2v_parsed_folder/$w2v_databasename.csv -output $w2v_vector_folder/"$w2v_databasename"_v"$l_vector"_n"$l_neg"_i"$l_iter"_csv.txt -size $l_vector -sample 1e-3 -negative $l_neg -hs 0 -binary 0 -cbow 1 -iter $l_iter -threads $w2v_threads
                # echo "==-=="
              fi
            done <<< "$iterations"
          fi
        done <<< "$negatives"
      fi
    done <<< "$vectors"
  fi
}


set -o verbose
source $1
echo $(run_word2vec $w2v_vectors $w2v_negatives $w2v_iterations $w2v_type)
