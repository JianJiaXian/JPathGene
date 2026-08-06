# Data preparation — JPathGene (feature-level image + gene fusion)

JPathGene operates on **pre-extracted feature-level CSVs** (no raw WSI
processing in this first version). Each cohort lives in its own directory under
the dataset root `data/` and must contain matched
histopathology image features and genetic/omics features, joined by
`patient_id`.

## Expected layout

```
data/
  FEATURE_CSV/            # default demo cohort (configs/feature_csv.yaml)
    image_features.csv
    genomic_features.csv  # gene_features.csv is also accepted
    labels.csv
    clinical.csv          # optional
    splits.csv            # optional
  TCGA_BRCA/              # real cohort (configs/tcga_brca.yaml)
    image_features.csv
    genomic_features.csv
    labels.csv
    clinical.csv
    splits.csv
  TCGA_LUAD/              # template (configs/tcga_luad.yaml) — not yet downloaded
    ...
```

## Required columns

`image_features.csv`
```
patient_id,img_0,img_1,...,img_D       # pre-extracted WSI / patch-mean features
```

`genomic_features.csv` (or `gene_features.csv`)
```
patient_id,gene_0_SYMBOL,gene_1_SYMBOL,...   # RNA-seq / mutation / CNV / pathway
```
(Plain `gene_0,gene_1,...` headers are also fine; the trailing `_SYMBOL` is used
only for interpretability and is optional.)

`labels.csv`
```
patient_id,label[,survival_time,survival_event]
```

`clinical.csv` (optional)
```
patient_id,age,stage,grade,...
```

`splits.csv` (optional — otherwise a deterministic stratified split is created)
```
patient_id,split        # split in {train,val,test}
```

## Join semantics

Only patients present in **all** of `image_features`, `genomic_features` and
`labels` are kept (inner join). Histology is typically available for fewer
patients than genomics, so the effective paired cohort can be smaller than the
genomic table — this is expected and reported by `check_feature_dataset.py`.

## Reference cohorts

* `FEATURE_CSV/` — a fully-aligned feature-level demo cohort (1.5k patients,
  64 image dims, 80 gene dims) used as the default so the whole pipeline is
  runnable end-to-end.
* `TCGA_BRCA/` — a real TCGA-BRCA cohort: pre-extracted ResNet WSI features
  (2048-d) for the patients with diagnostic slides, top-variance RNA-seq genes
  (200 named genes), a binary clinical endpoint, plus clinical and split files.
  After the inner join ~117 patients carry both modalities.

To build these cohorts from scratch (TCGA-BRCA):

1. Download `HiSeqV2` (RNA-seq) and `BRCA_clinicalMatrix` from UCSC Xena.
2. Select the top-K variance genes, z-score, write `genomic_features.csv`.
3. Extract patch features from diagnostic WSIs with a frozen encoder
   (ResNet50 / CTransPath), patient-mean pool, write `image_features.csv`.
4. Derive a binary label (e.g. survival/stage) into `labels.csv`.
5. Write a `splits.csv` (patient-level, stratified).

## TCGA-LUAD (future cohort)

Repeat the same procedure for LUAD into `TCGA_LUAD/` and run with
`configs/tcga_luad.yaml`. The pipeline is already wired for it.

## Sanity check

```bash
python scripts/check_feature_dataset.py --config configs/feature_csv.yaml
```

If the dataset root is missing the loader raises a clear `FileNotFoundError`
pointing back to this file — it never crashes silently or fabricates data.
