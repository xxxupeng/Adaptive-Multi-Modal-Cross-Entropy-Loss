from postprocess.disparity_regression import *

__disparity_regression__ = {
    "mean": mean_disparityregression,
    "SM": unimodal_disparityregression,
    "DM": unimodal_disparityregression_Dominant,
    "argmax": argmax_disparityregression,
}
