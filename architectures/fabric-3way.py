import keras
from keras.layers import Dense, Dropout, Input


def declare_model(input_dim, embedding_dim):
    input_s = Input(shape=(input_dim,), name="input_s")
    input_p = Input(shape=(input_dim,), name="input_p")
    input_o = Input(shape=(input_dim,), name="input_o")

    encoded_s = Dense(embedding_dim * 4, activation='relu', name="embed_s")(input_s)
    encoded_p = Dense(embedding_dim * 4, activation='relu', name="embed_p")(input_p)
    encoded_o = Dense(embedding_dim * 4, activation='relu', name="embed_o")(input_o)

    s_merge_p = keras.layers.concatenate([encoded_s, encoded_p], name="s_merge_p")
    p_merge_o = keras.layers.concatenate([encoded_p, encoded_o], name="p_merge_o")

    s_p_out = Dense(input_dim, activation='sigmoid', name="s_p_output")(s_merge_p)
    p_o_out = Dense(input_dim, activation='sigmoid', name="p_s_output")(p_merge_o)

    encoded_sp = Dense(embedding_dim * 4, activation='relu', name="encoded_sp")(s_merge_p)
    encoded_po = Dense(embedding_dim * 4, activation='relu', name="encoded_po")(p_merge_o)

    spo_merge = keras.layers.concatenate([encoded_sp, encoded_po], name="fact")

    encoded_spo = Dense(embedding_dim * 4, activation='relu', name="encoded_spo")(spo_merge)




