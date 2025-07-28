import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import allel
import scipy.stats as stats
from sklearn.decomposition import PCA
import base64

# Background image setup using base64
@st.cache_data
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64_image("genomeplant.jpg")  # Ensure this image exists in the same directory

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Genomic Trait Analysis for Pest and Drought Resistance")

# Upload files
vcf_file = st.file_uploader("Upload VCF File", type=["vcf"])
phenotype_file = st.file_uploader("Upload Phenotypic CSV File", type=["csv"])

if vcf_file and phenotype_file:
    snp_data = allel.read_vcf(vcf_file)
    phenotype_data = pd.read_csv(phenotype_file)

    geno = allel.GenotypeArray(snp_data['calldata/GT']).to_n_alt()
    samples = snp_data['samples']

    snp_df = pd.DataFrame(geno.T, columns=snp_data['variants/ID'])
    snp_df['Sample'] = samples
    full_data = pd.merge(phenotype_data, snp_df, on='Sample')

    st.subheader("GWAS Results")
    results = []
    traits = ['PestResistance', 'DroughtResistance']

    for trait in traits:
        y = full_data[trait].values
        for snp in snp_data['variants/ID']:
            X = sm.add_constant(full_data[snp].values)
            model = sm.OLS(y, X).fit()
            pval = model.pvalues[1]
            results.append({'Trait': trait, 'SNP': snp, 'P-Value': pval})

    results_df = pd.DataFrame(results)

    for trait in traits:
        trait_results = results_df[results_df['Trait'] == trait].sort_values('P-Value')
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(-np.log10(trait_results['P-Value'].values))
        ax.set_title(f'Manhattan Plot for {trait}')
        ax.set_xlabel('SNP Index')
        ax.set_ylabel('-log10(P-Value)')
        st.pyplot(fig)

    st.subheader("Genomic Selection")
    top_snps = results_df.nsmallest(50, 'P-Value')['SNP'].unique()
    X = full_data[top_snps]
    y = full_data['PestResistance']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100),
        'RidgeRegression': Ridge(alpha=1.0)
    }

    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=3, scoring='r2')
        st.write(f'{name} R2 score: {np.mean(scores):.3f}')

    st.subheader("QTL Mapping")
    qtl_results = []
    positions = snp_data['variants/POS']
    for snp, pos in zip(snp_data['variants/ID'], positions):
        slope, intercept, r_value, p_value, std_err = stats.linregress(full_data[snp], full_data['PestResistance'])
        qtl_results.append({'SNP': snp, 'Position': pos, 'P-Value': p_value, 'R2': r_value**2})

    qtl_df = pd.DataFrame(qtl_results)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(qtl_df['Position'], -np.log10(qtl_df['P-Value']))
    ax.set_title('QTL Mapping for Pest Resistance')
    ax.set_xlabel('Genomic Position')
    ax.set_ylabel('-log10(P-Value)')
    st.pyplot(fig)

    st.subheader("PCA Clustering")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=y, cmap='coolwarm')
    fig.colorbar(scatter, label='Pest Resistance')
    ax.set_title('PCA of Genomic Data Colored by Trait')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    st.pyplot(fig)

    candidate_snps = results_df[results_df['P-Value'] < 1e-5]['SNP'].unique()
    st.write("Top Candidate SNPs:", candidate_snps)

    st.subheader("Functional Annotation (Placeholder)")
    st.write("Use Biopython Entrez API to annotate SNPs - implementation placeholder")
