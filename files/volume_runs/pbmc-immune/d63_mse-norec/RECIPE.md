# d63_mse-norec

reconstruction likelihood: **mse**  |  CVAE recon: **OFF**

## ENCODER-path budget (what teaches the encoder)

- cohesion_gram: lambda=2.902454, share=25.0%
- cohesion_aff: lambda=5.073052, share=25.0%
- recon: lambda=0.0, share=0.0%
- gmkl: lambda=0.022276, share=50.0%

## PROTOTYPE-path budget (the only two terms touching prototypes)

- gmkl: share=50.0%
- proto_recon: share=50.0%

## routing

- cohesion: prototypes DETACHED (coh_proto_sg=1)
- proto_recon: form 3 (mix3), scores DETACHED, decoder LIVE
- community: OFF; nassoc/modularity/usage/swav/plain-KL: OFF

## outcome

- dead prototypes: 0
- sigma median: 0.9656
- collapse: {'median_nn_cos': 0.8055830001831055, 'frac_pairs_gt_0.99': 0.0, 'max_nn_cos': 0.8873748183250427}
