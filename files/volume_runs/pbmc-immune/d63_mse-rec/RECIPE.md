# d63_mse-rec

reconstruction likelihood: **mse**  |  CVAE recon: **ON**

## ENCODER-path budget (what teaches the encoder)

- cohesion_gram: lambda=2.902454, share=16.7%
- cohesion_aff: lambda=5.073052, share=16.7%
- recon: lambda=0.021126, share=33.3%
- gmkl: lambda=0.022276, share=33.3%

## PROTOTYPE-path budget (the only two terms touching prototypes)

- gmkl: share=50.0%
- proto_recon: share=50.0%

## routing

- cohesion: prototypes DETACHED (coh_proto_sg=1)
- proto_recon: form 3 (mix3), scores DETACHED, decoder LIVE
- community: OFF; nassoc/modularity/usage/swav/plain-KL: OFF

## outcome

- dead prototypes: 0
- sigma median: 0.8815
- collapse: {'median_nn_cos': 0.8268412947654724, 'frac_pairs_gt_0.99': 0.0, 'max_nn_cos': 0.9620459675788879}
