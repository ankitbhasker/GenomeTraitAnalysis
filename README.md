
<img width="1313" height="712" alt="image" src="https://github.com/user-attachments/assets/3e52887d-f6e2-41c1-8a22-8077ccabc19e" />

🌿 Genomic Trait Analysis Web Application
📌 Project Title:
Genomic Trait Analysis for Pest and Drought Resistance in Plants using VCF and Phenotypic Data

🧭 Project Objective:
The main goal of this application is to analyze genomic variations in plants and correlate them with traits such as pest resistance and drought tolerance. This enables breeders, researchers, and agronomists to:

Identify significant SNPs (Single Nucleotide Polymorphisms),

Predict desirable traits using genomic selection, and

Explore QTLs (Quantitative Trait Loci) through interactive visualization.

⚙️ Features of the App:
1. User-friendly Interface (Streamlit-based):
Interactive GUI to upload .vcf and .csv phenotype files.

Dynamic charts and model scores.

Background customization for a professional UI.

2. VCF File Handling:
Parses genotype information from standard .vcf (Variant Call Format) files using scikit-allel.

Converts genotype calls to numerical values (0, 1, 2 → homozygous ref, heterozygous, homozygous alt).

3. Phenotypic Data Integration:
Merges genotype data with phenotypic traits like PestResistance and DroughtResistance.

Ensures sample IDs match across datasets.

4. Genome-Wide Association Studies (GWAS):
Performs linear regression for each SNP against traits.

Displays Manhattan plots to highlight SNPs significantly associated with traits.

5. Genomic Selection (Machine Learning):
Uses Random Forest and Ridge Regression to predict traits from top SNPs.

Outputs R² scores via 5-fold cross-validation.

6. QTL Mapping:
Maps the genomic position of SNPs affecting traits using linear regression.

Displays results in scatter plots with -log10(p-value) to indicate strength of association.

7. Principal Component Analysis (PCA):
Reduces dimensionality of genotype data.

Plots genetic variation and clusters based on trait scores.

8. Candidate SNP Reporting:
Lists SNPs with high statistical significance (p < 1e-5).

Useful for marker-assisted selection and breeding.

9. Functional Annotation (Planned Feature):
Placeholder to later integrate Biopython’s Entrez for SNP annotation.

Future versions may fetch gene names, biological functions, and pathways.

📂 Input Files:
File Type	Description
.vcf	Contains genetic variants (SNPs) for each sample
.csv	Contains phenotypic trait values for each sample, including a Sample column to match

🧠 Technologies & Libraries:
Category	Tools Used
Web Framework	Streamlit
Data Handling	Pandas, NumPy
Statistics	Statsmodels, SciPy
ML Models	scikit-learn (RandomForest, Ridge)
Visualization	Matplotlib, Seaborn
Genomics	scikit-allel
PCA	scikit-learn
Styling	Base64 image for background

🌍 Significance to Human Kind:
Agriculture Improvement: Helps in developing climate-resilient crops by identifying drought-tolerant genotypes.

Pest Resistance: Accelerates the discovery of natural plant defenses, reducing dependency on pesticides.

Precision Breeding: Empowers breeders with data-driven decisions to select superior genotypes.

Genomic Medicine Parallel: Similar techniques apply to human genomic studies for trait-disease associations.

Food Security: Facilitates sustainable farming practices by promoting hardy crop varieties.

📈 Future Enhancements:
Full functional SNP annotation using NCBI API.

Add interactive SNP selection.

Support multiple phenotypic traits simultaneously.

Integrate expression data (RNA-Seq).

Export full analysis report in PDF.

Deploy on cloud platforms like Streamlit Cloud or AWS.
