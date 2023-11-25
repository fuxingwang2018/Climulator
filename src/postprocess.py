
from matplotlib import pyplot as plt

class PostProcess(object):

    def __init__(self):

        #self.y_pred = y_pred
        #self.X_test = X_test
        #self.y_test = y_test
        pass


    def plot_result(self, y_pred, X_test, y_test, path_figure):

        n = 50

        var_p = y_pred[n, :, :, 0]
        var_ref = y_test[n, :, :, 0]
        var_in = X_test[n, :, :, -1]

        fig, ax = plt.subplots(1, 3, figsize = (10, 4))

        ax[0].imshow(var_in)
        ax[1].imshow(var_p)
        ax[2].imshow(var_ref)
        fig.savefig(path_figure + "SRGAN_result.png")
        print('plot_result Done')


    def plot_input_data(self, var_low_res_gen, var_high_res, path_figure):

        n = 2000

        #var_low_res_test = var_low_res_gen[n, :, :, 0]
        #var_high_res_test = var_high_res[n, :, :, 0]
        var_low_res_test = var_low_res_gen[n, ]
        var_high_res_test = var_high_res[n, ]

        fig, ax = plt.subplots(1, 2, figsize = (10, 4))

        ax[0].imshow(var_low_res_test)
        ax[1].imshow(var_high_res_test)

        fig.savefig(path_figure + "SRGAN_input.png")
        print('plot_input_data Done')

        #n = 2000

        #var_low_res_test = var_low_res_gen[n, :, :, 0]
        #var_high_res_test = var_high_res[n, :, :, 0]

        #fig, ax = plt.subplots(1, 2, figsize = (10, 4))

        #ax[0].imshow(var_low_res_test)
        #ax[1].imshow(var_high_res_test)
        #fig.savefig(path_figure + "SRGAN_input.png")
