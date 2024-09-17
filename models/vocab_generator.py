import re
import numpy as np
import pandas as pd
from utils.rxn import *
from utils.molecule import *
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

mission_list = ["BH", "Suzuki", "AT", "SNAR"]

# size of word vector
size = 128

if "BH" in mission_list:
    train_set = list()
    vocab_dict = dict()
    seq_len = list()
    # BH_HTE
    BH_HTE_df = pd.read_excel("../data/BH_HTE/BH_HTE_data.xlsx")
    for i in range(BH_HTE_df.shape[0]):
        text = get_Buchwald_RxnSmi(BH_HTE_df.iloc[i, :])
        train_set.append(smi_tokenizer(text))
        seq_len.append(len(smi_tokenizer(text)))
    print("Maximum sequence length is:" % max(np.array(seq_len)))

    # sequence length distribution
    plt.figure(dpi=500)
    sns.kdeplot(seq_len, fill=True)
    plt.xlabel("Sequence Length")
    plt.ylabel("Counting")
    plt.yticks([])
    plt.title("Buchwald-Hartwig Sequence Distribution")
    plt.tight_layout()
    plt.savefig("./Buchwald-Hartwig Sequence Distribution.png")

    # generate vocab file
    print("The length of train set is: %d" % len(train_set))
    word_id = dict()
    word_vec = list()
    model = Word2Vec(train_set, vector_size=size, window=30, min_count=5, epochs=20, sg=1)
    for i, w in enumerate(model.wv.index_to_key):
        vocab_dict[w] = model.wv[w]
        # record for evaluation
        word_id[w] = i
        word_vec.append(model.wv[w])
    vocab_dict_to_txt(vocab_dict, rxn_name="BH")
    print("There are %d atoms in the dict" % len(vocab_dict))

    # Evaluation
    X_reduced = PCA(n_components=2).fit_transform(np.array(word_vec))
    plt.figure(dpi=500)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], color="black")

    for w in word_id.keys():
        xy = X_reduced[word_id[w], :]
        plt.scatter(xy[0], xy[1], color="r")
        plt.text(xy[0], xy[1], w, color="b")

    plt.title("Buchwald-Hartwig Vocab Distribution")
    plt.tight_layout()
    plt.savefig("./Buchwald-Hartwig Vocab Distribution.png" )

if "Suzuki" in mission_list:
    train_set = list()
    vocab_dict = dict()
    seq_len = list()
    # Suzuki_HTE
    Suzuki_HTE_df = pd.read_excel("../data/Suzuki_HTE/Suzuki_HTE_data.xlsx")
    for i in range(Suzuki_HTE_df.shape[0]):
        text = get_Suzuki_RxnSmi(Suzuki_HTE_df.iloc[i, :])
        train_set.append(smi_tokenizer(text))
        seq_len.append(len(smi_tokenizer(text)))
    print("Maximum sequence length is:" % max(np.array(seq_len)))

    # sequence length distribution
    plt.figure(dpi=500)
    sns.kdeplot(seq_len, fill=True)
    plt.xlabel("Sequence Length")
    plt.ylabel("Counting")
    plt.yticks([])
    plt.title("Suzuki-Miyaura Sequence Distribution")
    plt.tight_layout()
    plt.savefig("./Suzuki-Miyaura Sequence Distribution.png")

    # generate vocab file
    print("The length of train set is: %d" % len(train_set))
    word_id = dict()
    word_vec = list()
    model = Word2Vec(train_set, vector_size=size, window=45, min_count=3, epochs=25, sg=1)
    for i, w in enumerate(model.wv.index_to_key):
        vocab_dict[w] = model.wv[w]
        # record for evaluation
        word_id[w] = i
        word_vec.append(model.wv[w])
    vocab_dict_to_txt(vocab_dict, rxn_name="Suzuki")
    print("There are %d atoms in the dict" % len(vocab_dict))

    # Evaluation
    X_reduced = PCA(n_components=2).fit_transform(np.array(word_vec))
    plt.figure(dpi=500)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], color="black")

    for w in word_id.keys():
        xy = X_reduced[word_id[w], :]
        plt.scatter(xy[0], xy[1], color="r")
        plt.text(xy[0], xy[1], w, color="b")

    plt.title("Suzuki-Miyaura Vocab Distribution")
    plt.tight_layout()
    plt.savefig("./Suzuki-Miyaura Vocab Distribution.png" )

if "AT" in mission_list:
    train_set = list()
    vocab_dict = dict()
    seq_len = list()
    # Asymmetric_Thiol_Addition
    AT_df = pd.read_csv("../data/AT/Asymmetric_Thiol_Addition.csv")
    for i in range(AT_df.shape[0]):
        text = get_AT_RxnSmi(AT_df.iloc[i, :])
        train_set.append(smi_tokenizer(text))
        seq_len.append(len(smi_tokenizer(text)))
    print("Maximum sequence length is:" % max(np.array(seq_len)))

    # sequence length distribution
    plt.figure(dpi=500)
    sns.kdeplot(seq_len, fill=True)
    plt.xlabel("Sequence Length")
    plt.ylabel("Counting")
    plt.yticks([])
    plt.title("Asymmetric Thiol Sequence Distribution")
    plt.tight_layout()
    plt.savefig("./Asymmetric Thiol Sequence Distribution.png")

    # generate vocab file
    print("The length of train set is: %d" % len(train_set))
    word_id = dict()
    word_vec = list()
    model = Word2Vec(train_set, vector_size=size, window=80, min_count=3, epochs=20, sg=1)
    for i, w in enumerate(model.wv.index_to_key):
        vocab_dict[w] = model.wv[w]
        # record for evaluation
        word_id[w] = i
        word_vec.append(model.wv[w])
    vocab_dict_to_txt(vocab_dict, rxn_name="AT")
    print("There are %d atoms in the dict" % len(vocab_dict))

    # Evaluation
    X_reduced = PCA(n_components=2).fit_transform(np.array(word_vec))
    plt.figure(dpi=500)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], color="black")

    for w in word_id.keys():
        xy = X_reduced[word_id[w], :]
        plt.scatter(xy[0], xy[1], color="r")
        plt.text(xy[0], xy[1], w, color="b")

    plt.title("Asymmetric Thiol Vocab Distribution")
    plt.tight_layout()
    plt.savefig("./Asymmetric Thiol Vocab Distribution.png" )

if "SNAR" in mission_list:
    train_set = list()
    vocab_dict = dict()
    seq_len = list()
    # Asymmetric_Thiol_Addition
    SNAR_df = pd.read_excel("../data/SNAR/SNAR_data.xlsx")
    for i in range(SNAR_df.shape[0]):
        text = get_SNAR_RxnSmi(SNAR_df.iloc[i, :])
        train_set.append(smi_tokenizer(text))
        seq_len.append(len(smi_tokenizer(text)))
    print("Maximum sequence length is:" % max(np.array(seq_len)))

    # sequence length distribution
    plt.figure(dpi=500)
    sns.kdeplot(seq_len, fill=True)
    plt.xlabel("Sequence Length")
    plt.ylabel("Counting")
    plt.yticks([])
    plt.title("S$_N$Ar Sequence Distribution")
    plt.tight_layout()
    plt.savefig("./SNAR Sequence Distribution.png")

    # generate vocab file
    print("The length of train set is: %d" % len(train_set))
    word_id = dict()
    word_vec = list()
    model = Word2Vec(train_set, vector_size=size, window=20, min_count=2, epochs=50, sg=1)
    for i, w in enumerate(model.wv.index_to_key):
        vocab_dict[w] = model.wv[w]
        # record for evaluation
        word_id[w] = i
        word_vec.append(model.wv[w])
    vocab_dict_to_txt(vocab_dict, rxn_name="SNAR")
    print("There are %d atoms in the dict" % len(vocab_dict))

    # Evaluation
    X_reduced = PCA(n_components=2).fit_transform(np.array(word_vec))
    plt.figure(dpi=500)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], color="black")

    for w in word_id.keys():
        xy = X_reduced[word_id[w], :]
        plt.scatter(xy[0], xy[1], color="r")
        plt.text(xy[0], xy[1], w, color="b")

    plt.title("S$_N$Ar Vocab Distribution")
    plt.tight_layout()
    plt.savefig("./SNAR Vocab Distribution.png")