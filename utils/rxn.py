"""
This Module contains some tools to convert the formation of molecules
"""
import pandas as pd
import numpy as np
import torch
from utils.molecule import *
from rxnfp.transformer_fingerprints import *
from drfp import DrfpEncoder


def RxnSmi_to_tensor(RxnSmi, maxlen_, victor_size, file):
    vocab_dict = vocab_txt_to_dict(file) # get vocab_dict

    atoms = RxnSmi.split(" ")
    embedding_matrix = np.zeros((maxlen_, victor_size))

    for i in range(len(atoms)):
      atom = atoms[i]
      embedding_glove_vector = vocab_dict[atom] if atom in vocab_dict else None
      if embedding_glove_vector is not None:
        embedding_matrix[i] = embedding_glove_vector
      else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_matrix[i] = unk_vec

    return torch.tensor(embedding_matrix, dtype=torch.float32)

def get_Buchwald_RxnSmi(BH_HTE_df):
    base = smi_tokenizer(BH_HTE_df.loc["base_smiles"])
    ligand = smi_tokenizer(BH_HTE_df.loc["ligand_smiles"])
    aryl_halide = smi_tokenizer(BH_HTE_df.loc["aryl_halide_smiles"])
    additive = smi_tokenizer(BH_HTE_df.loc["additive_smiles"])
    product = smi_tokenizer(BH_HTE_df.loc["product_smiles"])

    text = smi_tokenizer("CC1=CC=C(N)C=C1") + ["."] + aryl_halide + [">"] + additive + ["."] + base + ["."] + ligand + [
        ">"] + product

    return "".join(text)

def get_Suzuki_RxnSmi(Suzuki_HTE_df):
    reactant1 = Suzuki_HTE_df.loc["Reactant_1_Name"]
    reactant2 = Suzuki_HTE_df.loc["Reactant_2_Name"]
    cat = Suzuki_HTE_df.loc["Catalyst_1_Short_Hand"]
    ligand = Suzuki_HTE_df.loc["Ligand_Short_Hand"]
    base = Suzuki_HTE_df.loc["Reagent_1_Short_Hand"]
    sol = Suzuki_HTE_df.loc["Solvent_1_Short_Hand"]
    product = "CC1=CC=C2C(C=NN2C2CCCCO2)=C1C1C=C2C=CC=NC2=CC=1"

    if pd.isnull(ligand) and pd.isnull(base):
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + [">"] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(sol) + [">"] + smi_tokenizer(product)
    if pd.isnull(ligand) and pd.isnull(base) == False:
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + [">"] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(base) + ["."] + smi_tokenizer(sol) + [">"] + smi_tokenizer(product)
    if pd.isnull(ligand) == False and pd.isnull(base):
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + [">"] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(ligand) + ["."] + smi_tokenizer(sol) + [">"] + smi_tokenizer(product)
    if pd.isnull(ligand) == False and pd.isnull(base) == False:
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + [">"] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(ligand) + ["."] + smi_tokenizer(base) + ["."] + smi_tokenizer(sol) + [
                   ">"] + smi_tokenizer(product)

    return "".join(text)

def get_AT_RxnSmi(AT_df):
    reactant1 = AT_df.loc["Imine"]
    reactant2 = AT_df.loc["Thiol"]
    cat = AT_df.loc["Catalyst"]
    product = AT_df.loc["product"]

    text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + [">"] + smi_tokenizer(cat) + [">"] + \
           smi_tokenizer(product)

    return "".join(text)

def get_SNAR_RxnSmi(SNAR_df):
    reactant1 = SNAR_df.loc["Substrate SMILES"]
    reactant2 = SNAR_df.loc["Nucleophile SMILES"]
    sols = SNAR_df.loc["Solvent"].split(".")

    product = SNAR_df.loc["Product SMILES"]

    text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + [">"]
    for sol in sols:
        text = text + smi_tokenizer(sol) + ["."]
    text = text[:-1] + [">"] + smi_tokenizer(product)

    return "".join(text)

def get_ELN_RxnSmi(ELN_df):
    reactant1 = ELN_df.loc["reactant1"]
    reactant2 = ELN_df.loc["reactant2"]
    product = ELN_df.loc["product"]
    cats = ELN_df.loc["catalyst"].split(".")
    sol = ELN_df.loc["solvent"]
    base = ELN_df.loc["base"]

    text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + [">"]
    for cat in cats:
        text = text + smi_tokenizer(cat) + ["."]
    text = text[:-1] + smi_tokenizer(sol) + smi_tokenizer(base) + [">"] + smi_tokenizer(product)

    return "".join(text)

def get_Buchwald_rxnfp(BH_HTE_df):
    base = smi_tokenizer(BH_HTE_df.loc["base_smiles"])
    ligand = smi_tokenizer(BH_HTE_df.loc["ligand_smiles"])
    aryl_halide = smi_tokenizer(BH_HTE_df.loc["aryl_halide_smiles"])
    additive = smi_tokenizer(BH_HTE_df.loc["additive_smiles"])
    product = smi_tokenizer(BH_HTE_df.loc["product_smiles"])

    text = smi_tokenizer("CC1=CC=C(N)C=C1") + ["."] + aryl_halide + ["."] + additive + ["."] + base + ["."] + ligand + [
        ">>"] + product
    text = "".join(text)

    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    rxnfp = rxnfp_generator.convert(text)

    return rxnfp

def get_Buchwald_drfp(BH_HTE_df):
    base = smi_tokenizer(BH_HTE_df.loc["base_smiles"])
    ligand = smi_tokenizer(BH_HTE_df.loc["ligand_smiles"])
    aryl_halide = smi_tokenizer(BH_HTE_df.loc["aryl_halide_smiles"])
    additive = smi_tokenizer(BH_HTE_df.loc["additive_smiles"])
    product = smi_tokenizer(BH_HTE_df.loc["product_smiles"])

    text = smi_tokenizer("CC1=CC=C(N)C=C1") + ["."] + aryl_halide + ["."] + additive + ["."] + base + ["."] + ligand + [
        ">>"] + product
    text = "".join(text)

    drfp = DrfpEncoder.encode(text)[0]

    return drfp

def get_Suzuki_rxnfp(Suzuki_HTE_df):
    reactant1 = Suzuki_HTE_df.loc["Reactant_1_Name"]
    reactant2 = Suzuki_HTE_df.loc["Reactant_2_Name"]
    cat = Suzuki_HTE_df.loc["Catalyst_1_Short_Hand"]
    ligand = Suzuki_HTE_df.loc["Ligand_Short_Hand"]
    base = Suzuki_HTE_df.loc["Reagent_1_Short_Hand"]
    sol = Suzuki_HTE_df.loc["Solvent_1_Short_Hand"]
    product = "CC1=CC=C2C(C=NN2C2CCCCO2)=C1C1C=C2C=CC=NC2=CC=1"

    if pd.isnull(ligand) and pd.isnull(base):
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(sol) + [">>"] + smi_tokenizer(product)
    if pd.isnull(ligand) and pd.isnull(base) == False:
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(base) + ["."] + smi_tokenizer(sol) + [">>"] + smi_tokenizer(product)
    if pd.isnull(ligand) == False and pd.isnull(base):
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(ligand) + ["."] + smi_tokenizer(sol) + [">>"] + smi_tokenizer(product)
    if pd.isnull(ligand) == False and pd.isnull(base) == False:
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(ligand) + ["."] + smi_tokenizer(base) + ["."] + smi_tokenizer(sol) + [
                   ">>"] + smi_tokenizer(product)
    text = "".join(text)

    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    rxnfp = rxnfp_generator.convert(text)

    return rxnfp

def get_Suzuki_drfp(Suzuki_HTE_df):
    reactant1 = Suzuki_HTE_df.loc["Reactant_1_Name"]
    reactant2 = Suzuki_HTE_df.loc["Reactant_2_Name"]
    cat = Suzuki_HTE_df.loc["Catalyst_1_Short_Hand"]
    ligand = Suzuki_HTE_df.loc["Ligand_Short_Hand"]
    base = Suzuki_HTE_df.loc["Reagent_1_Short_Hand"]
    sol = Suzuki_HTE_df.loc["Solvent_1_Short_Hand"]
    product = "CC1=CC=C2C(C=NN2C2CCCCO2)=C1C1C=C2C=CC=NC2=CC=1"

    if pd.isnull(ligand) and pd.isnull(base):
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(sol) + [">>"] + smi_tokenizer(product)
    if pd.isnull(ligand) and pd.isnull(base) == False:
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(base) + ["."] + smi_tokenizer(sol) + [">>"] + smi_tokenizer(product)
    if pd.isnull(ligand) == False and pd.isnull(base):
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(ligand) + ["."] + smi_tokenizer(sol) + [">>"] + smi_tokenizer(product)
    if pd.isnull(ligand) == False and pd.isnull(base) == False:
        text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."] + smi_tokenizer(cat) + [
            "."] + smi_tokenizer(ligand) + ["."] + smi_tokenizer(base) + ["."] + smi_tokenizer(sol) + [
                   ">>"] + smi_tokenizer(product)
    text = "".join(text)

    drfp = DrfpEncoder.encode(text)[0]

    return drfp

def get_AT_rxnfp(AT_df):
    reactant1 = AT_df.loc["Imine"]
    reactant2 = AT_df.loc["Thiol"]
    cat = AT_df.loc["Catalyst"]
    product = AT_df.loc["product"]

    text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."] + smi_tokenizer(cat) + [">>"] + \
           smi_tokenizer(product)
    text = "".join(text)

    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    rxnfp = rxnfp_generator.convert(text)

    return rxnfp

def get_AT_drfp(AT_df):
    reactant1 = AT_df.loc["Imine"]
    reactant2 = AT_df.loc["Thiol"]
    cat = AT_df.loc["Catalyst"]
    product = AT_df.loc["product"]

    text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."] + smi_tokenizer(cat) + [">>"] + \
           smi_tokenizer(product)
    text = "".join(text)

    drfp = DrfpEncoder.encode(text)[0]

    return drfp

def get_SNAR_rxnfp(SNAR_df):
    reactant1 = SNAR_df.loc["Substrate SMILES"]
    reactant2 = SNAR_df.loc["Nucleophile SMILES"]
    sols = SNAR_df.loc["Solvent"].split(".")
    product = SNAR_df.loc["Product SMILES"]

    text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."]
    for sol in sols:
        text = text + smi_tokenizer(sol) + ["."]
    text = text[:-1] + [">>"] + smi_tokenizer(product)
    text = "".join(text)

    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    rxnfp = rxnfp_generator.convert(text)

    return rxnfp

def get_SNAR_drfp(SNAR_df):
    reactant1 = SNAR_df.loc["Substrate SMILES"]
    reactant2 = SNAR_df.loc["Nucleophile SMILES"]
    sols = SNAR_df.loc["Solvent"].split(".")
    product = SNAR_df.loc["Product SMILES"]

    text = smi_tokenizer(reactant1) + ["."] + smi_tokenizer(reactant2) + ["."]
    for sol in sols:
        text = text + smi_tokenizer(sol) + ["."]
    text = text[:-1] + [">>"] + smi_tokenizer(product)
    text = "".join(text)

    drfp = DrfpEncoder.encode(text)[0]

    return drfp

def read_rxnfp(arr_str):
    arr_str = arr_str[1:]
    arr_str = arr_str[:-1]
    arr_str = arr_str.split(", ")
    for i in range(len(arr_str)):
        arr_str[i] = float(arr_str[i])
    return np.array(arr_str).astype(np.float16)

def read_drfp(arr_str):
    arr_str = arr_str[1:]
    arr_str = arr_str[:-1]
    arr_str = arr_str.split(" ")
    for i in range(len(arr_str)):
        if "\n" in arr_str[i]:
            arr_str[i] = arr_str[i][0]
        arr_str[i] = float(arr_str[i])
    return np.array(arr_str).astype(np.float16)
