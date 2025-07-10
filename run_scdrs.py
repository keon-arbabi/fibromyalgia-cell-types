import os
import scdrs
import scanpy as sc
from utils import Timer, run
from single_cell import SingleCell

file_neurons = 'single-cell/Human-Brain-Atlas/All_Neurons.h5ad'
if not os.path.exists(file_neurons):
    run(f'wget https://datasets.cellxgene.cziscience.com/'
        f'9ecb4ba-b033-4a93-b794-05e262dc1f59.h5ad '
        f'-O {file_neurons}')
file_nonneurons = 'single-cell/Human-Brain-Atlas/All_Non_Neurons.h5ad'
if not os.path.exists(file_nonneurons):
    run(f'wget https://datasets.cellxgene.cziscience.com/'
        f'9082ad42-b5ba-449f-ad10-1b988ac79eaa.h5ad '
        f'-O {file_nonneurons}')

sc_neurons = SingleCell(file_neurons) 
sc_nonneurons = SingleCell(file_nonneurons) 



df_gs = scdrs.util.load_gs(
    'source/fibromyalgia-cell-types/input/scdrs/fibro.gs')
adata = scdrs.util.load_h5ad(
    'single-cell/SEAAD/Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad',
     flag_filter_data=False, flag_raw_count=False)

scdrs.preprocess(adata)
gene_list = df_gs['fibro'][0]
gene_weight = df_gs['fibro'][1]
df_res = scdrs.score_cell(adata, gene_list, gene_weight=gene_weight)
