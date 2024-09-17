"""
This Module contains some tools to convert the formation of molecules

"""
# smi to fp
import torch
from torch_geometric.data import Data
import numpy as np

def AtomCharityEmbed(atom):
    C = str(atom.GetChiralTag())
    OH = [0 for i in range(3)] # [Whether it is Chiral or not, R, S]
    # No Chiral
    if C == "CHI_UNSPECIFIED":
        return OH
    # R
    if C == "CHI_TETRAHEDRAL_CW":
        OH[0] = 1
        OH[1] = 1
        return OH
    # S
    if C == "CHI_TETRAHEDRAL_CCW":
        OH[0] = 1
        OH[2] = 1
        return OH

    return OH

def AtomHybridizationEmbed(atom):
    HBZ = str(atom.GetHybridization())
    OH = [0 for i in range(6)] # [sp, sp2, sp3, sp3d, sp3d2, other]
    DICT = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
    if HBZ == "SP":
        OH[0] = 1
        return OH
    if HBZ == "SP2":
        OH[1] = 1
        return OH
    if HBZ == "SP3":
        OH[2] = 1
        return OH
    if HBZ == "SP3D":
        OH[3] = 1
        return OH
    if HBZ == "SP3D2":
        OH[4] = 1
        return OH
    if HBZ not in DICT:
        OH[5] = 1
        return OH

def smi_to_graph(smi, NoFeature=False):
    """
    :param smi:
    :return:
        :num: number of atoms
        :feature: feature of nodes, include AtomicNumber & FormalCharge
        :edge_index: edge information
    """
    # Get num
    mol = Chem.MolFromSmiles(smi)
    num = mol.GetNumAtoms()
    # Get label
    feature = list()
    atoms = mol.GetAtoms()
    # Get Charge contribute
    AllChem.ComputeGasteigerCharges(mol)

    for atom in atoms:
        if NoFeature == False:
            atom_feature = [
                int(atom.GetAtomicNum()), # Atomic Number,
                int(atom.GetFormalCharge()), # Formal Charge,
                int(atom.GetTotalNumHs()), # Number of connected H,
                int(atom.GetExplicitValence()), # Explicit Valence,
                int(atom.GetDegree()), # Degree of an atom
                int(atom.GetIsAromatic()), # is aromatic or not
                int(atom.IsInRing()), # is in ring or not
                float(atom.GetProp("_GasteigerCharge"))  # Gasteiger Charge Contribution
            ]
            # atom_feature += AtomCharityEmbed(atom) # Atom Charity
            # atom_feature += AtomHybridizationEmbed(atom) # Atom Hybridization

            feature.append(atom_feature)
        else:
            atom_feature = [
                int(0),
            ]
            feature.append(atom_feature)

    # Get edge_index
    us = list()
    vs = list()
    bonds = mol.GetBonds()
    for bond in bonds:
        u = bond.GetBeginAtom().GetIdx()
        v = bond.GetEndAtom().GetIdx()
        us.append(u)
        vs.append(v)
        us.append(v)
        vs.append(u)
    edge_index = [us, vs]

    return num, feature, edge_index

import torch_geometric.transforms as T
def smis_to_graph(smis, NoFeature=False):
    temp_num = 0
    g_feature = list()
    g_us = list()
    g_vs = list()
    for smi in smis:
        num, feature, edge_index = smi_to_graph(smi, NoFeature=NoFeature)
        for atom in feature:
            g_feature.append(atom)
        for u in edge_index[0]:
            g_us.append(u + temp_num)
        for v in edge_index[1]:
            g_vs.append(v + temp_num)
        temp_num += num
    g_edge_index = [g_us, g_vs]

    x = torch.tensor(g_feature, dtype=torch.float32)
    edge_index = torch.tensor(g_edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    data = T.NormalizeFeatures()(data)
    data = T.AddSelfLoops()(data) # Self Loop

    return data

# SMILES Tokenizer
def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule
    """
    import re
    pattern = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens

# F:Format conversion: vocab_list & vocab_dict
def vocab_dict_to_txt(dict, rxn_name):
    with open("../utils/%s_vocab.txt" % rxn_name, mode="w", encoding="utf-8") as txt:
        for key in dict.keys():
            txt.write("%s\t%s\t\t" % (key, dict[key]))
    txt.close()

def vocab_txt_to_dict(file):
    vocab_dict = dict()
    txt = open(file, mode="r", encoding="utf-8")
    vocab = txt.read().replace("\n", "").split("\t\t")
    vocab = vocab[:-1] # remove the last one, which is ""

    for pair in vocab:
        pair = pair.split("\t")
        vec = pair[1][1:-1].replace("  ", " ").split(" ") # delete "[", "]" and split by " "
        while "" in vec:
            vec.remove("") # remove null element
        vocab_dict[pair[0]] = np.array(vec).astype(float)

    return vocab_dict

# HighLight of important atoms in a molecule
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from PIL import Image as PILImage
import io
import matplotlib.pyplot as plt

def red_cmap(x):
    """Red color map"""
    x = max(0, min(1, x))  # Ensure the value is within [0, 1]
    # Red for higher value
    return 1.0, 1.0 - x, 1.0 -x

def color_bond(bond, saliency, color_fn):
    begin = saliency[bond.GetBeginAtomIdx()]
    end = saliency[bond.GetEndAtomIdx()]
    return color_fn((begin + end) / 2)

def moltopng(mol, atom_colors, bond_colors, bondwidth=2, molSize=(600,600), kekulize=True):
    MolSize = molSize
    Bondwidth = bondwidth
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DCairo(*MolSize)
    drawer.drawOptions().useBWAtomPalette()
    drawer.drawOptions().bondLineWidth = Bondwidth
    drawer.drawOptions().highlightBondWidth = 2
    drawer.DrawMolecule(
        mc,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBonds=list(bond_colors.keys()),
        highlightBondColors=bond_colors,
        highlightAtomRadii={i: .4 for i in range(len(atom_colors))}
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

def highlight_mol(mol, weight, molsize=(600, 600), bondwidth=6, bond=False, atom_indices=None, bond_indices=None):
    Molsize = molsize
    Bondwidth = bondwidth
    # Default to highlighting all atoms if no specific indices are given
    if atom_indices is None:
        atom_indices = list(range(mol.GetNumAtoms()))

    # Generate colors only for specified atoms
    atom_colors = {i: red_cmap(w) for i, w in zip(atom_indices, [weight[i] for i in atom_indices])}

    if bond:
        bondlist = mol.GetBonds()
        if bond_indices is None:
            bond_colors = {i: color_bond(bond, weight, red_cmap) for i, bond in enumerate(bondlist)}
        else:
            bondlist = [bondlist[i] for i in bond_indices]
            bond_colors = {i: color_bond(bond, weight, red_cmap) for i, bond in zip(bond_indices, bondlist)}
    else:
        bond_colors = {}

    return moltopng(mol, atom_colors, bond_colors, Bondwidth, Molsize)

def mols_with_colorbar(images, subplot_size, fig_size=(8, 6)):
    # Create a Figure
    fig = plt.figure(figsize=fig_size, dpi=500)

    # Create a GridSpec with extra space for the colorbar, associated with the figure
    gs = gridspec.GridSpec(subplot_size[0], subplot_size[1]+1, figure=fig,
                           width_ratios=[1]*subplot_size[1] + [0.06])

    # Create subplots using the GridSpec
    axs = [fig.add_subplot(gs[i, j]) for i in range(subplot_size[0]) for j in range(subplot_size[1])]

    # Loop over the images and subplots
    for ax, image in zip(axs, images):
        # Convert image data to numpy array
        img_data = np.array(PILImage.open(io.BytesIO(image)))
        # Display the image on the subplot
        ax.imshow(img_data)
        ax.axis('off')  # Turn off axis if you don't want to see it

    # Create a ScalarMappable for the colorbar
    cmap = cm.ScalarMappable(cmap=cm.Reds)
    cmap.set_array([0, 1])  # Set the data range for the colorbar

    # Add the colorbar to the last column of the GridSpec
    cbar = fig.colorbar(cmap, cax=fig.add_subplot(gs[:, subplot_size[1]]), orientation='vertical')
    cbar.ax.tick_params(width=2)  # Increase the width of the ticks
    cbar.outline.set_linewidth(2) # Increase the width of the outline

    # Adjust the layout manually or use constrained_layout
    fig.set_constrained_layout(True)  # Use this instead of tight_layout

    return fig
