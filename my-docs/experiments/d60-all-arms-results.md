
### Cohesion (aff)

| metric | epoch0_wayp | epoch0_leid | d59_base_broken | d59_protorecon | d60_leid_noc | d60_wayp_noc | d60_wayp_comm | scpoli_leiden | seacell_ref |
|---|---|---|---|---|---|---|---|---|---|
| coh_aff|train|all | 0.0898 | 0.0806 | 0.1349 | 0.1365 | 0.1310 | 0.1298 | 0.1343 | 0.0591 | 0.1014 |
| coh_aff|test|all | 0.0880 | 0.0784 | 0.0816 | 0.0760 | 0.0853 | 0.0818 | 0.0848 | 0.0535 | 0.1014 |
| coh_aff|train|rare | 0.1616 | 0.1263 | 0.2465 | 0.2484 | 0.1670 | 0.2295 | 0.2476 | 0.0688 | 0.2753 |
| coh_aff|test|rare | 0.1540 | 0.1262 | 0.1650 | 0.1594 | 0.1376 | 0.1624 | 0.1657 | 0.0674 | 0.2773 |
| coh_aff|train|common | 0.0875 | 0.0791 | 0.1313 | 0.1329 | 0.1298 | 0.1266 | 0.1307 | 0.0588 | 0.0958 |
| coh_aff|test|common | 0.0859 | 0.0769 | 0.0790 | 0.0733 | 0.0837 | 0.0792 | 0.0822 | 0.0531 | 0.0958 |

### Cohesion (gram)

| metric | epoch0_wayp | epoch0_leid | d59_base_broken | d59_protorecon | d60_leid_noc | d60_wayp_noc | d60_wayp_comm | scpoli_leiden | seacell_ref |
|---|---|---|---|---|---|---|---|---|---|
| coh_gram|train|all | 0.1689 | 0.1534 | 0.2356 | 0.2376 | 0.2243 | 0.2266 | 0.2358 | 0.1106 | 0.2077 |
| coh_gram|test|all | 0.1664 | 0.1508 | 0.1592 | 0.1474 | 0.1629 | 0.1581 | 0.1665 | 0.1049 | 0.2080 |
| coh_gram|test|rare | 0.3169 | 0.2672 | 0.3178 | 0.3126 | 0.2988 | 0.3262 | 0.3192 | 0.1521 | 0.4981 |
| coh_gram|test|common | 0.1616 | 0.1470 | 0.1542 | 0.1421 | 0.1586 | 0.1528 | 0.1617 | 0.1034 | 0.1988 |

### Within-batch purity

| metric | epoch0_wayp | epoch0_leid | d59_base_broken | d59_protorecon | d60_leid_noc | d60_wayp_noc | d60_wayp_comm | scpoli_leiden | seacell_ref |
|---|---|---|---|---|---|---|---|---|---|
| within|train|all | 0.8883 | 0.8790 | 0.8990 | 0.9010 | 0.8937 | 0.8957 | 0.9022 | 0.8675 | 0.8913 |
| within|test|all | 0.8927 | 0.8886 | 0.8542 | 0.8387 | 0.8722 | 0.8619 | 0.8706 | 0.8708 | 0.8912 |
| within|test|rare | 0.7437 | 0.5700 | 0.6316 | 0.6117 | 0.6269 | 0.6546 | 0.6378 | 0.3465 | 0.8150 |
| within|test|common | 0.8976 | 0.8991 | 0.8613 | 0.8459 | 0.8805 | 0.8687 | 0.8778 | 0.8885 | 0.8936 |

### Cross-batch agreement

| metric | epoch0_wayp | epoch0_leid | d59_base_broken | d59_protorecon | d60_leid_noc | d60_wayp_noc | d60_wayp_comm | scpoli_leiden | seacell_ref |
|---|---|---|---|---|---|---|---|---|---|
| cross|train|all | 0.9630 | 0.9577 | 0.8555 | 0.8654 | 0.8956 | 0.8648 | 0.8828 | 0.9665 | 0.9793 |
| cross|test|all | 0.9541 | 0.9545 | 0.8849 | 0.8815 | 0.9244 | 0.9197 | 0.9218 | 0.9827 | 0.9865 |
| cross|test|rare | 0.9296 | 0.9550 | 0.4526 | 0.4628 | 0.9154 | 0.7990 | 0.6757 | 0.9653 | 1.0000 |
| cross|test|common | 0.9549 | 0.9545 | 0.8986 | 0.8947 | 0.9247 | 0.9237 | 0.9294 | 0.9833 | 0.9861 |

### Pooled purity

| metric | epoch0_wayp | epoch0_leid | d59_base_broken | d59_protorecon | d60_leid_noc | d60_wayp_noc | d60_wayp_comm | scpoli_leiden | seacell_ref |
|---|---|---|---|---|---|---|---|---|---|
| pooled|train|all | 0.8708 | 0.8599 | 0.7834 | 0.7869 | 0.8199 | 0.7959 | 0.8041 | 0.8533 | 0.8814 |
| pooled|test|all | 0.8680 | 0.8630 | 0.7678 | 0.7524 | 0.8160 | 0.7943 | 0.8071 | 0.8565 | 0.8802 |
| pooled|test|rare | 0.6570 | 0.5362 | 0.1942 | 0.1845 | 0.6039 | 0.5000 | 0.4461 | 0.3092 | 0.8107 |
| pooled|test|common | 0.8747 | 0.8734 | 0.7861 | 0.7704 | 0.8228 | 0.8037 | 0.8185 | 0.8740 | 0.8824 |

### Rare metrics (test)

| metric | epoch0_wayp | epoch0_leid | d59_base_broken | d59_protorecon | d60_leid_noc | d60_wayp_noc | d60_wayp_comm | scpoli_leiden | seacell_ref |
|---|---|---|---|---|---|---|---|---|---|
| rare_homog|test | 0.7812 | 0.7584 | 0.6074 | 0.6118 | 0.7190 | 0.6355 | 0.7199 | 0.6398 | 0.8747 |
| rare_f1|test | 0.6946 | 0.8790 | 0.6977 | 0.7054 | 0.7958 | 0.7110 | 0.7856 | 0.6662 | 0.8927 |
| rare_cov|test | 0.7667 | 1.0000 | 0.9333 | 1.0000 | 1.0000 | 0.9333 | 1.0000 | 0.7667 | 0.9500 |

### Rare metrics (train)

| metric | epoch0_wayp | epoch0_leid | d59_base_broken | d59_protorecon | d60_leid_noc | d60_wayp_noc | d60_wayp_comm | scpoli_leiden | seacell_ref |
|---|---|---|---|---|---|---|---|---|---|
| rare_homog|train | 0.7374 | 0.7261 | 0.6964 | 0.7514 | 0.7181 | 0.7262 | 0.7244 | 0.6209 | 0.8526 |
| rare_f1|train | 0.6878 | 0.8455 | 0.7022 | 0.7916 | 0.7741 | 0.7096 | 0.6881 | 0.6214 | 0.8769 |
| rare_cov|train | 0.7667 | 1.0000 | 0.9333 | 1.0000 | 1.0000 | 0.9333 | 0.8667 | 0.7667 | 1.0000 |