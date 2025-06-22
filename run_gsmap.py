import os
import gzip
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import random
from utils import run, p_to_abs_z, read_csv_delim_whitespace

workdir = 'projects/rrg-wainberg/karbabi/fibromyalgia_cell_types'

if not os.path.exists(f'{workdir}/gsMap_resource'):
    run(f'wget https://yanglab.westlake.edu.cn/data/gsMap/gsMap_resource.tar.gz '
        f'-P {workdir}')
    run(f'tar -xvzf {workdir}/gsMap_resource.tar.gz -C {workdir}')
    run(f'rm {workdir}/gsMap_resource.tar.gz')

if not os.path.exists(f'{workdir}/gsMap_example_data'):
    run(f'wget https://yanglab.westlake.edu.cn/data/gsMap/gsMap_example_data.tar.gz '
        f'-P {workdir}')
    run(f'tar -xvzf {workdir}/gsMap_example_data.tar.gz -C {workdir}')
    run(f'rm {workdir}/gsMap_example_data.tar.gz')

if not os.path.exists(f'{workdir}/out'):
    os.makedirs(f'{workdir}/out')
if not os.path.exists(f'{workdir}/figures'):
    os.makedirs(f'{workdir}/figures')

raw_gwas_file = f'{workdir}/gsMap_example_data/GWAS/' \
    'fibromyalgia_all_updated_rsID_EUR.meta.gz'
formatted_gwas_file = f'{workdir}/gsMap_example_data/GWAS/' \
    'fibromyalgia_all_updated_rsID_EUR.sumstats.gz'

if not os.path.exists(formatted_gwas_file):
    gwas_df = read_csv_delim_whitespace(raw_gwas_file)
    abs_z = p_to_abs_z(gwas_df['P'])
    or_col_numeric = gwas_df['OR'].cast(pl.Float64, strict=False)
    sign = pl.when(or_col_numeric > 1).then(1) \
             .when(or_col_numeric < 1).then(-1) \
             .otherwise(0)
    z_score = abs_z * sign
    formatted_df = gwas_df.select(
        pl.col('SNP'),
        pl.col('A1'),
        pl.col('A2'),
        z_score.alias('Z'),
        pl.col('NEFF_sum').alias('N')
    ).drop_nulls()
    csv_string = formatted_df.write_csv(separator='\t')
    with gzip.open(formatted_gwas_file, "wt") as f:
        f.write(csv_string)

run(f'''
    gsmap quick_mode \
    --workdir '{workdir}/out' \
    --homolog_file '{workdir}/gsMap_resource/homologs/mouse_human_homologs.txt' \
    --sample_name 'E16.5_E1S1.MOSTA' \
    --gsMap_resource_dir '{workdir}/gsMap_resource' \
    --hdf5_path '{workdir}/gsMap_example_data/ST/E16.5_E1S1.MOSTA.h5ad' \
    --annotation 'annotation' \
    --data_layer 'count' \
    --sumstats_file '{formatted_gwas_file}' \
    --trait_name 'Fibromyalgia'
''')

color_dict = {
    "Adipose tissue": "#6e2ee6",
    "Adrenal gland": "#c00250",
    "Bone": "#113913",
    "Brain": "#ef833a",
    "Cartilage": "#35596b",
    "Cartilage primordium": "#3bb54c",
    "Cavity": "#dddddf",
    "Choroid plexus": "#bd39dc",
    "Connective tissue": "#0bd4b4",
    "Dorsal root ganglion": "#b84b11",
    "Epidermis": "#046df4",
    "GI tract": "#5c5ba9",
    "Heart": "#d3245a",
    "Inner ear": "#03fff3",
    "Jaw and tooth": "#f061fa",
    "Kidney": "#61cfe5",
    "Liver": "#ca23b2",
    "Lung": "#7dc136",
    "Meninges": "#e0ca44",
    "Mucosal epithelium": "#307dd3",
    "Muscle": "#ae1041",
    "Smooth muscle": "#fc5252",
    "Spinal cord": "#fad5bb",
    "Submandibular gland": "#ab33e6",
    "Sympathetic nerve": "#cd5a0c"
}

sample_name = 'E16.5_E1S1.MOSTA'
trait_name = 'Fibromyalgia'

gsmap_plot_file = (
    f'{workdir}/out/{sample_name}/report/{trait_name}/gsMap_plot/'
    f'{sample_name}_{trait_name}_gsMap_plot.csv'
)
cauchy_results_file = (
    f'{workdir}/out/{sample_name}/cauchy_combination/'
    f'{sample_name}_{trait_name}.Cauchy.csv.gz'
)

if os.path.exists(gsmap_plot_file) and os.path.exists(cauchy_results_file):
    df = pl.read_csv(gsmap_plot_file).to_pandas()
    plot_df = pl.read_csv(cauchy_results_file)\
        .sort('p_cauchy')\
        .with_columns((-pl.col('p_cauchy').log10())
                      .alias('-log10(p_cauchy)'))

    fig = plt.figure(figsize=(22, 8))
    fig.set_facecolor('white')
    gs_main = gridspec.GridSpec(
        1, 2, width_ratios=[2, 0.6], wspace=0.1, figure=fig
    )
    gs_spatial = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_main[0], wspace=0.01
    )
    ax1 = fig.add_subplot(gs_spatial[0])
    ax2 = fig.add_subplot(gs_spatial[1])
    ax3 = fig.add_subplot(gs_main[1])

    sns.scatterplot(
        data=df, x='sx', y='sy', hue='logp', palette='magma',
        s=5, linewidth=0, ax=ax1, legend=False
    )
    ax1.set_facecolor('black')
    ax1.set_aspect('equal', adjustable='box')
    ax1.axis('off')
    ax1.set_title(trait_name, color='black', fontsize=16)

    norm = plt.Normalize(df['logp'].min(), df['logp'].max())
    sm = plt.cm.ScalarMappable(cmap="magma", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.13, 0.08, 0.15, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'$-\log_{10}(\mathrm{P\ value})$', color='black', size=12)
    cbar.ax.tick_params(colors='black')
    cbar.outline.set_edgecolor('black')

    annos = sorted(df['annotation'].unique())
    color_map = {anno: color_dict[anno] for anno in annos}
    sns.scatterplot(
        data=df, x='sx', y='sy', hue='annotation', hue_order=annos,
        palette=color_map, s=5, linewidth=0, ax=ax2, legend=False
    )
    ax2.set_facecolor('black')
    ax2.set_aspect('equal', adjustable='box')
    ax2.axis('off')
    ax2.set_title('E16.5', color='black', fontsize=16)

    plot_df_pd = plot_df.to_pandas()
    sns.barplot(
        data=plot_df_pd, x='-log10(p_cauchy)', y='annotation',
        palette=color_map, ax=ax3, hue='annotation', dodge=False,
        legend=False, order=plot_df_pd['annotation']
    )
    ax3.set_title(f'gsMap Cauchy P-values for {trait_name}')
    ax3.set_xlabel('-log10(p-value)')
    ax3.set_ylabel('')

    path_svg = (f'{workdir}/figures/'
                f'{sample_name}_{trait_name}_spatial_combined.svg')
    path_png = (f'{workdir}/figures/'
                f'{sample_name}_{trait_name}_spatial_combined.png')
    plt.savefig(path_svg, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(path_png, bbox_inches='tight', pad_inches=0.1)