
from matplotlib import pyplot as plt
import numpy as np

class PostProcess(object):

    def __init__(self):

        #self.y_pred = y_pred
        #self.X_test = X_test
        #self.y_test = y_test
        pass


    def plot_result(self, y_pred, X_test, y_test, path_figure, X_name, y_name):

        n = 15 #50
        nt_Xtest, nx_Xtest, ny_Xtest, nv_Xtest = np.shape(X_test)

        var_p = y_pred[n, :, :, 0]
        var_ref = y_test[n, :, :, 0]
        #var_in = X_test[n, :, :, -1]

        nrow, ncol = 3, (nv_Xtest + 2)//3 + 1 
        fig, ax = plt.subplots(nrow, ncol, figsize = (12, 8))
        ax = ax.flatten()

        for ivar in range(nv_Xtest):
            ax[ivar].imshow(X_test[n, :, :, ivar])
            ax[ivar].set_title('LR: ' + X_name[ivar])

        ax[nv_Xtest].imshow(var_p)
        ax[nv_Xtest].set_title('Pred: ' + y_name[0]) 

        ax[nv_Xtest + 1].imshow(var_ref)
        ax[nv_Xtest + 1].set_title('Ref: ' + y_name[0]) 
         
        """
        ax[0].imshow(var_in)
        ax[0].set_title('Input') 
        """

        fig.savefig(path_figure + "SRGAN_result.png")
        print('plot_result Done')


    def plot_input_data(self, var_low_res_dict, var_high_res_dict, path_figure):

        n = 16 #2000
        nvar_low_res, nvar_high_res  = len(var_low_res_dict), len(var_high_res_dict)
        nvar = nvar_low_res + nvar_high_res
        nrow, ncol = 3, nvar//3 + 1 #max(nvar_high_res, nvar_low_res)
        fig, ax = plt.subplots(nrow, ncol, figsize = (12, 8))
        ax = ax.flatten()

        i = 0
        for var_low_res_key, var_low_res_values in var_low_res_dict.items():
            ax[i].imshow(var_low_res_values[n, ])
            ax[i].set_title('LR: ' + str(var_low_res_key)) 
            i += 1

        for var_high_res_key, var_high_res_values in var_high_res_dict.items():
            ax[i].imshow(var_high_res_values[n, ])
            ax[i].set_title('HR: ' + str(var_high_res_key)) 
            i += 1
            
        fig.savefig(path_figure + "SRGAN_input.png")
        print('plot_input_data Done')


    def plot_lines(self, data_dict, path_figure):

        fig, ax1 = plt.subplots()
        gen_loss, disc_loss = ['gen_loss',  'val_gen_loss'], ['disc_loss', 'val_disc_loss']
        linestyle_def = {'gen_loss': '-', 'val_gen_loss': '--', 'disc_loss': '-', 'val_disc_loss':'--'}
        color = 'tab:red'
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Generator Loss', color=color)

        for key in gen_loss:
            ax1.plot(data_dict[key], linestyle = linestyle_def[key], label=key, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Discriminator Loss', color=color)
        for key in disc_loss:
            ax2.plot(data_dict[key], linestyle = linestyle_def[key], label=key, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')

        plt.grid(visible=True, which='major', axis='both', linestyle=':', linewidth=1 )
        plt.title('Vriation of loss function with steps')

        fig.tight_layout()  
        plt.savefig(path_figure)
        plt.close()

"""
# Example dictionary
data = {
    'gen_loss': [1, 2, 3, 4, 5],
    'val_gen_loss': [3, 4, 5, 6, 8],
    'disc_loss': [100, 200, 300, 450, 500],
    'val_disc_loss': [200, 300, 400, 500, 700]
}

# Path to save the plot
save_path = 'plot.png'

# Plot lines from dictionary and save plot to PNG
postp = PostProcess() 
postp.plot_lines(data, save_path)
"""



