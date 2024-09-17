import pandas as pd
import numpy as np
from utils.rxn import *
from tqdm import tqdm
np.set_printoptions(threshold=np.inf)

# import data
BH_df = pd.read_excel("../data/BH_HTE/BH_HTE_data.xlsx")
Suzuki_df = pd.read_excel("../data/Suzuki_HTE/Suzuki_HTE_data.xlsx")
AT_df = pd.read_csv("../data/AT/Asymmetric_Thiol_Addition.csv")
SNAR_df = pd.read_excel("../data/SNAR/SNAR_data.xlsx")

# Generate RXNFP and DRFP
# BH
BH_rxnfp = list()
BH_drfp = list()
for i in tqdm(range(BH_df.shape[0])):
    rxn = BH_df.loc[i]
    BH_rxnfp.append(str(get_Buchwald_rxnfp(rxn)))
    BH_drfp.append(str(get_Buchwald_drfp(rxn)))
BH_rxnfp = pd.DataFrame(BH_rxnfp, columns=["rxnfp"])
BH_drfp = pd.DataFrame(BH_drfp, columns=["drfp"])
BH_df = pd.concat([BH_df, BH_rxnfp, BH_drfp], axis=1)
BH_df.to_excel("../data/BH_HTE/BH_fp.xlsx")

# Suzuki
Suzuki_rxnfp = list()
Suzuki_drfp = list()
for i in tqdm(range(Suzuki_df.shape[0])):
    rxn = Suzuki_df.loc[i]
    Suzuki_rxnfp.append(str(get_Suzuki_rxnfp(rxn)))
    Suzuki_drfp.append(str(get_Suzuki_drfp(rxn)))
Suzuki_rxnfp = pd.DataFrame(Suzuki_rxnfp, columns=["rxnfp"])
Suzuki_drfp = pd.DataFrame(Suzuki_drfp, columns=["drfp"])
Suzuki_df = pd.concat([Suzuki_df, Suzuki_rxnfp, Suzuki_drfp], axis=1)
Suzuki_df.to_excel("../data/Suzuki_HTE/Suzuki_fp.xlsx")

# AT
AT_rxnfp = list()
AT_drfp = list()
for i in tqdm(range(AT_df.shape[0])):
    rxn = AT_df.loc[i]
    AT_rxnfp.append(str(get_AT_rxnfp(rxn)))
    AT_drfp.append(str(get_AT_drfp(rxn)))
AT_rxnfp = pd.DataFrame(AT_rxnfp, columns=["rxnfp"])
AT_drfp = pd.DataFrame(AT_drfp, columns=["drfp"])
AT_df = pd.concat([AT_df, AT_rxnfp, AT_drfp], axis=1)
AT_df.to_excel("../data/AT/AT_fp.xlsx")

# SNAR
SNAR_rxnfp = list()
SNAR_drfp = list()
for i in tqdm(range(SNAR_df.shape[0])):
    rxn = SNAR_df.loc[i]
    SNAR_rxnfp.append(str(get_SNAR_rxnfp(rxn)))
    SNAR_drfp.append(str(get_SNAR_drfp(rxn)))
SNAR_rxnfp = pd.DataFrame(SNAR_rxnfp, columns=["rxnfp"])
SNAR_drfp = pd.DataFrame(SNAR_drfp, columns=["drfp"])
SNAR_df = pd.concat([SNAR_df, SNAR_rxnfp, SNAR_drfp], axis=1)
SNAR_df.to_excel("../data/SNAR/SNAR_fp.xlsx")

