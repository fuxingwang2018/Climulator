
from matplotlib import pyplot as plt

class PostProcess(object):

    def __init__(self, pr_pred, X_test, y_test):
        self.pr_pred = pr_pred
        self.X_test = X_test
        self.y_test = y_test

    def plot_result(self, path_figure):

        n = 50

        var_p = self.pr_pred[n, :,:, 0]
        var_ref = self.y_test[n, :,:, 0]
        var_in = self.X_test[n, :,:, 0]

        fig, ax = plt.subplots(1, 3, figsize = (10, 4))

        ax[0].imshow(var_in)
        ax[1].imshow(var_p)
        ax[2].imshow(var_ref)
        fig.savefig(path_figure + "SRGAN_result.png")

