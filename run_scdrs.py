import os
import scdrs
import pandas as pd
import pickle as pkl
import scanpy as sc
import matplotlib.pyplot as plt
from utils import run

data_dir = 'single-cell/Human-Brain-Atlas/'
input_dir = 'projects/rrg-wainberg/karbabi/fibromyalgia-cell-types/input/scdrs'
out_dir = 'projects/rrg-wainberg/karbabi/fibromyalgia-cell-types/output/scdrs'
fig_dir = 'projects/rrg-wainberg/karbabi/fibromyalgia-cell-types/figures/scdrs'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# https://www.science.org/doi/10.1126/science.add7046
files = {
    'All_Non_Neurons': f'{data_dir}/All_Non_Neurons.h5ad',  
    'All_Neurons': f'{data_dir}/All_Neurons.h5ad'
}
urls = {
    'All_Non_Neurons': 'https://datasets.cellxgene.cziscience.com/' 
        '9082ad42-b5ba-449f-ad10-1b988ac79eaa.h5ad',
    'All_Neurons': 'https://datasets.cellxgene.cziscience.com/' 
        'f9ecb4ba-b033-4a93-b794-05e262dc1f59.h5ad'
}
for name, file in files.items():
    if not os.path.exists(file):
        run(f'wget {urls[name]} -O {file}')

df_gs = scdrs.util.load_gs(f'{input_dir}/fibro.gs')
adatas = {}

for name, file in files.items():
    print(name)
    print(f'Loading {name}...')
    adatas[name] = scdrs.util.load_h5ad(
        file, flag_filter_data=False, flag_raw_count=True)
    
    print(f'Preprocessing {name}...')    
    sc.pp.highly_variable_genes(
        adatas[name], n_top_genes=2000, batch_key='donor_id')
    sc.pp.pca(adatas[name])
    sc.pp.neighbors(adatas[name])
    
    file_uns = f'{out_dir}/{name}_uns.pkl'
    if os.path.exists(file_uns):
        print(f'Loading metadata {name}...')
        adatas[name].uns = pkl.load(open(file_uns, 'rb'))
    else:
        print(f'Computing metadata {name}...')
        df_cov = pd.DataFrame({
            'const': 1,
            'n_genes': adatas[name].obs['total_genes'],
            'sex_male': (adatas[name].obs['sex'] == 'male').astype(int),
            'age': adatas[name].obs['development_stage'].str.extract(r'(\d+)')
                .squeeze().astype(int)
        }, index=adatas[name].obs.index)

        df_cov_dummy = pd.DataFrame(        
            index=adatas[name].obs.index,
            data={'const': 1})

        scdrs.preprocess(
            adatas[name], 
            cov=df_cov_dummy, 
            n_chunk=50 if name == 'All_Neurons' else 20)
        pkl.dump(adatas[name].uns, open(file_uns, 'wb'))

dfs_res, dfs_stats = {}, {}

for name in files.keys():
    file_res = f'{out_dir}/{name}_res.pkl'
    if os.path.exists(file_res):
        print(f'Loading scDRS results {name}...')
        dfs_res[name] = pkl.load(open(file_res, 'rb'))
    else:
        print(f'Computing scDRS results {name}...')
        dfs_res[name] = scdrs.score_cell(
            adatas[name], 
            gene_list=df_gs['fibro'][0],
            gene_weight=df_gs['fibro'][1],
            ctrl_match_key='mean_var',  
            n_ctrl=1000,
            n_genebin=200,
            weight_opt='vs',
            return_ctrl_raw_score=False,
            return_ctrl_norm_score=True,
            verbose=True
        )
        pkl.dump(dfs_res[name], open(file_res, 'wb'))

    file_stats = f'{out_dir}/{name}_stats.csv'
    if os.path.exists(file_stats):
        print(f'Loading downstream group analysis {name}...')
        dfs_stats[name] = pd.read_csv(file_stats, index_col=0)
    else:
        print(f'Computing downstream group analysis {name}...')
        dfs_stats[name] = scdrs.method.downstream_group_analysis(
            adata=adatas[name], 
            df_full_score=dfs_res[name],
            group_cols=['supercluster_term'],
        )['supercluster_term']
        dfs_stats[name].to_csv(file_stats)

for name in files.keys():
    print(name)
    adatas[name].obsm['X_umap'] = adatas[name].obsm['X_UMAP']
    adatas[name].obs['fibro_norm_score'] = dfs_res[name]['norm_score']
    sc.set_figure_params(figsize=[5, 5], dpi=300)
    sc.pl.umap(
        adatas[name],
        color="supercluster_term",
    )
    plt.savefig(
        f'{fig_dir}/{name}_umap_supercluster_term.png', 
        bbox_inches='tight')
    sc.pl.umap(
        adatas[name],
        color='fibro_norm_score',
        color_map="RdBu_r",
        vmin=-5,
        vmax=5,
    )
    plt.savefig(
        f'{fig_dir}/{name}_umap_norm_score.png', 
        bbox_inches='tight')
    plt.figure()
    scdrs.util.plot_group_stats(
        dict_df_stats={'fibro': dfs_stats[name]},
    )
    plt.savefig(
        f'{fig_dir}/{name}_group_stats.png',
        bbox_inches='tight'
    )
    plt.close()

