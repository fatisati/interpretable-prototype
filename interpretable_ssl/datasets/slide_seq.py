from interpretable_ssl.datasets.spatial import *
import squidpy as sq


class SlideSeq:
    def __init__(self):
        adata = sq.datasets.slideseqv2()
        