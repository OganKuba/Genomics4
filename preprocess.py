from pathlib import Path
import pandas as pd, numpy as np, h5py
from cyvcf2 import VCF
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[0]
DATA = ROOT / "data"
CNT  = DATA/"geuvadis/E-GEUV-1/processed/gene_counts.txt"
VCF_PATH = DATA/"1kg_sv/sv_geuvadis_only.vcf.gz"   # ← ścieżka do VCF
OUT  = DATA/"preprocessed"; OUT.mkdir(parents=True, exist_ok=True)

# --- 4.1  Counts → TPM -----------------------------------------------------
df = pd.read_csv(CNT, sep="\t", comment="#")
gene_len_kb = df["Length"] / 1e3
expr_raw = df.set_index("Geneid").drop(columns=["Chr","Start","End","Strand","Length"]).T
expr_raw.index = expr_raw.index.str.extract(r'(HG\d{5})')[0]
print("expr_raw.index (clean):", expr_raw.index.tolist())
rpk = expr_raw.div(gene_len_kb.values, axis=1)
tpm = rpk.div(rpk.sum(axis=1), axis=0) * 1e6
expr = np.log2(tpm + 1)
mask = (tpm > 1).sum(axis=0) >= int(0.2 * tpm.shape[0])
expr = expr.loc[:, mask]

# --- 4.2  Genotypy SV ------------------------------------------------------
v = VCF(str(VCF_PATH))          # ← poprawiona linijka
samples = v.samples
geno, sv_ids = [], []

for var in tqdm(v, desc="SV"):
    geno.append([ (a+b if a>=0 and b>=0 else 0) for a,b,*_ in var.genotypes ])
    sv_ids.append(var.ID)

geno = pd.DataFrame(np.array(geno).T, index=samples, columns=sv_ids)
print("geno.index:", geno.index.tolist())

# --- 4.3  Synchronizacja -----------------------------------------------
common = expr.index.intersection(geno.index)
expr = expr.loc[common].sort_index()
geno = geno.loc[common].sort_index()

# --- 4.4  Zapis HDF5 ----------------------------------------------------
h5 = OUT/"geuvadis_sv_expr.h5"
with h5py.File(h5, "w") as f:
    f.create_dataset("expression", data=expr.to_numpy(), compression="gzip")
    f.create_dataset("expr_genes", data=np.array(expr.columns.astype(str).values, dtype="S"))
    f.create_dataset("sv_gt",      data=geno.to_numpy(), compression="gzip")
    f.create_dataset("sv_ids",     data=np.array(geno.columns.astype(str).values, dtype="S"))
    f.create_dataset("samples",    data=np.array(common.astype(str).values, dtype="S20"))

print(f"✔  zapisano {h5}  ({expr.shape[0]} próbki, {expr.shape[1]} genów, {geno.shape[1]} SV)")
print(expr.index.tolist())
print(geno.index.tolist())
