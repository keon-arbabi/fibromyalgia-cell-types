import os
import scdrs
import pickle as pkl
import scanpy as sc
import pandas as pd
from utils import run
from single_cell import SingleCell 

data_dir = 'single-cell/Human-Brain-Atlas/'
input_dir = 'projects/rrg-wainberg/karbabi/fibromyalgia-cell-types/input/scdrs'
out_dir = 'projects/rrg-wainberg/karbabi/fibromyalgia-cell-types/output/scdrs'
os.makedirs(out_dir, exist_ok=True)

# https://www.science.org/doi/10.1126/science.add7046
files = {
    'All_Neurons': f'{data_dir}/All_Neurons.h5ad',
    'All_Non_Neurons': f'{data_dir}/All_Non_Neurons.h5ad'
}
urls = {
    'All_Neurons': 'https://datasets.cellxgene.cziscience.com/' 
        'f9ecb4ba-b033-4a93-b794-05e262dc1f59.h5ad',
    'All_Non_Neurons': 'https://datasets.cellxgene.cziscience.com/' 
        '9082ad42-b5ba-449f-ad10-1b988ac79eaa.h5ad'
}
for name, file in files.items():
    if not os.path.exists(file):
        run(f'wget {urls[name]} -O {file}')

df_gs = scdrs.util.load_gs(f'{input_dir}/fibro.gs')

adatas = {}

for name, file in files.items():
    print(name)
    adatas[name] = SingleCell(file)\
        .with_uns({'QCed': True})\
        .hvg().normalize().to_scanpy()

    file_uns = f'{out_dir}/{name}_uns.pkl'
    if os.path.exists(file_uns):
        adatas[name].uns = pkl.load(open(file_uns, 'rb'))
    else:
        df_cov = pd.DataFrame({
            'const': 1,
            'n_genes': adatas[name].obs['total_genes'],
            'sex_male': (adatas[name].obs['sex'] == 'male').astype(int),
            'age': adatas[name].obs['development_stage'].str.extract(r'(\d+)')
                .squeeze().astype(int)
        }, index=adatas[name].obs.index)

        # Dummy cov for triggering n_chunk logic 
        df_cov_dummy = pd.DataFrame(
            index=adatas[name].obs.index,
            data={'const': 1}
        )
        # No cov correction 
        scdrs.preprocess(adatas[name], cov=df_cov_dummy, n_chunk=50)
        pkl.dump(adatas[name].uns, open(file_uns, 'wb'))

dfs_res, dfs_stats = {}, {}

for name in files.keys():
    print(name)
    file_res = f'{out_dir}/{name}_res.pkl'
    if os.path.exists(file_res):
        dfs_res[name] = pkl.load(open(file_res, 'rb'))
    else:
        dfs_res[name] = scdrs.score_cell(
            adatas[name], 
            gene_list=df_gs['fibro'][0],
            gene_weight=df_gs['fibro'][1],
            ctrl_match_key='mean_var',  
            n_ctrl=1000,
            weight_opt='vs',
            return_ctrl_raw_score=False,
            return_ctrl_norm_score=True,
            verbose=False
        )
        pkl.dump(dfs_res[name], open(file_res, 'wb'))

    file_stats = f'{out_dir}/{name}_stats.pkl'
    if os.path.exists(file_stats):
        dfs_stats[name] = pkl.load(open(file_stats, 'rb'))
    else:
        dfs_stats[name] = scdrs.method.downstream_group_analysis(
            adata=adatas[name],
            df_full_score=dfs_res[name],
            group_cols=['supercluster_term'],
        )['supercluster_term']
        pkl.dump(dfs_stats[name], open(file_stats, 'wb'))

