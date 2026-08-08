# kBET / iLISI results — metacell-composition (Part A) and standard per-cell embedding (Part B)

Source notebook: `notebooks/kbet_ilisi_metacell.ipynb`

## Part A — metacell-composition (own group assignment as neighborhood)

| dataset | method | n_metacells | metacell_ilisi_mean | metacell_ilisi_weighted_mean | kbet_rejection_rate_by_mc | kbet_rejection_rate_by_cell | kbet_n_tested | kbet_n_excluded_small |
|---|---|---|---|---|---|---|---|---|
| pancreas | scProto | 219 | 1.579 | 2.919 | 0.973 | 0.999 | 110 | 109 |
| pancreas | SEACells (PCA) | 220 | 1.184 | 1.180 | 1.000 | 1.000 | 220 | 0 |
| pancreas | SEACells (scPoli (Stage-1)) | 220 | 2.960 | 3.051 | 0.982 | 0.998 | 220 | 0 |
| pancreas | Leiden (scPoli (Stage-1)) | 220 | 3.055 | 3.242 | 0.982 | 0.992 | 220 | 0 |
| lung | scProto | 298 | 1.891 | 3.827 | 0.949 | 0.999 | 156 | 142 |
| lung | SEACells (PCA) | 300 | 1.541 | 1.490 | 1.000 | 1.000 | 300 | 0 |
| lung | SEACells (scPoli (Stage-1)) | 300 | 3.172 | 3.336 | 1.000 | 1.000 | 300 | 0 |
| lung | Leiden (scPoli (Stage-1)) | 300 | 2.980 | 3.739 | 0.997 | 0.999 | 296 | 4 |
| pbmc-immune | scProto | 294 | 1.317 | 2.689 | 0.897 | 0.998 | 97 | 197 |
| pbmc-immune | SEACells (PCA) | 300 | 1.122 | 1.150 | 1.000 | 1.000 | 300 | 0 |
| pbmc-immune | SEACells (scPoli (Stage-1)) | 300 | 1.699 | 2.135 | 0.997 | 0.998 | 300 | 0 |
| pbmc-immune | Leiden (scPoli (Stage-1)) | 88 | 2.185 | 2.187 | 0.989 | 0.997 | 88 | 0 |

## Part B — standard per-cell embedding (scib_metrics.benchmark.Benchmarker, full default battery)

| dataset | method | Isolated labels | KMeans NMI | KMeans ARI | Silhouette label | cLISI | BRAS | iLISI | KBET | Graph connectivity | PCR comparison | Batch correction | Bio conservation | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pancreas | PCA (uncorrected, SEACells space) | 0.662942 | 0.619377 | 0.379469 | 0.54067 | 1.0 | 0.462921 | 0.000433 | 0.160427 | 0.719823 | 0.0 | 0.268721 | 0.640492 | 0.491783 |
| pancreas | scPoli (Stage-1) | 0.658021 | 0.737152 | 0.442318 | 0.632435 | 1.0 | 0.64388 | 0.168048 | 0.233493 | 0.923999 | 0.679833 | 0.529851 | 0.693985 | 0.628331 |
| pancreas | scProto (embedding) | 0.73417 | 0.639917 | 0.371988 | 0.600504 | 1.0 | 0.432995 | 0.046283 | 0.238665 | 0.800092 | 0.317978 | 0.367203 | 0.669316 | 0.548471 |
| lung | PCA (uncorrected, SEACells space) | 0.650876 | 0.669018 | 0.445872 | 0.570517 | 1.0 | 0.472952 | 0.001986 | 0.213284 | 0.729164 | 0.0 | 0.283477 | 0.667257 | 0.513745 |
| lung | scPoli (Stage-1) | 0.750681 | 0.647938 | 0.495837 | 0.582856 | 0.998488 | 0.670057 | 0.087789 | 0.317912 | 0.900696 | 0.838553 | 0.563001 | 0.69516 | 0.642296 |
| lung | scProto (embedding) | 0.647813 | 0.585438 | 0.354035 | 0.562055 | 0.999492 | 0.518007 | 0.059787 | 0.286334 | 0.872912 | 0.512282 | 0.449865 | 0.629767 | 0.557806 |
| pbmc-immune | PCA (uncorrected, SEACells space) | 0.599962 | 0.608178 | 0.412937 | 0.536031 | 0.996537 | 0.389445 | 0.0 | 0.216919 | 0.640025 | 0.0 | 0.249278 | 0.630729 | 0.478149 |
| pbmc-immune | scPoli (Stage-1) | 0.616273 | 0.646025 | 0.355114 | 0.572647 | 0.995895 | 0.612265 | 0.173941 | 0.247683 | 0.892511 | 0.957515 | 0.576783 | 0.637191 | 0.613027 |
| pbmc-immune | scProto (embedding) | 0.637683 | 0.684215 | 0.442114 | 0.583913 | 0.996731 | 0.557335 | 0.109466 | 0.22923 | 0.838378 | 0.914209 | 0.529723 | 0.668931 | 0.613248 |
