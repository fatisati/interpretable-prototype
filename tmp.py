to remove batch effect this is what i do:
    
train epoch:
    for each batch
        iter throgh data
            x1, x2 pos pairs
            compute loss
            loss.backward
            
    update weights
