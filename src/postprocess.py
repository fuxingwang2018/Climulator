
from matplotlib import pyplot as plt
import numpy as np

class PostProcess(object):

    def __init__(self):

        #self.y_pred = y_pred
        #self.X_test = X_test
        #self.y_test = y_test
        pass


    def plot_result(self, y_pred, X_test, y_test, path_figure, X_name, y_name):

        n = [5, 15]
        nt_Xtest, nx_Xtest, ny_Xtest, nv_Xtest = np.shape(X_test)
        nrow, ncol = 3, (nv_Xtest + 2)//3 + 1 

        for i_step in n:

            var_p = y_pred[i_step, :, :, 0]
            var_ref = y_test[i_step, :, :, 0]
            #var_in = X_test[i, :, :, -1]

            fig, ax = plt.subplots(nrow, ncol, figsize = (12, 8))
            ax = ax.flatten()

            for ivar in range(nv_Xtest):
                ax[ivar].imshow(X_test[i_step, :, :, ivar])
                ax[ivar].set_title('LR: ' + X_name[ivar])

            ax[nv_Xtest].imshow(var_ref)
            ax[nv_Xtest].set_title('Ref: ' + y_name[0]) 

            ax[nv_Xtest + 1].imshow(var_p)
            ax[nv_Xtest + 1].set_title('Pred: ' + y_name[0]) 
         
            fig.savefig(path_figure + f"SRGAN_result_step_{i_step}.png")

        print('plot_result Done')
 

    def plot_input_data(self, var_low_res_dict, var_high_res_dict, path_figure):

        n = [5, 15]
        nvar_low_res, nvar_high_res  = len(var_low_res_dict), len(var_high_res_dict)
        nvar = nvar_low_res + nvar_high_res
        nrow, ncol = 3, nvar//3 + 1 #max(nvar_high_res, nvar_low_res)

        for i_step in n:
            fig, ax = plt.subplots(nrow, ncol, figsize = (12, 8))
            ax = ax.flatten()

            i = 0
            for var_low_res_key, var_low_res_values in var_low_res_dict.items():
                ax[i].imshow(var_low_res_values[i_step, ])
                ax[i].set_title('LR: ' + str(var_low_res_key)) 
                i += 1

            for var_high_res_key, var_high_res_values in var_high_res_dict.items():
                ax[i].imshow(var_high_res_values[i_step, ])
                ax[i].set_title('HR: ' + str(var_high_res_key)) 
                i += 1
            
            fig.savefig(path_figure + f"SRGAN_input_step_{i_step}.png")
        print('plot_input_data Done')


    def plot_gan_history(self, data_dict, path_figure):

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
        plt.title('Variation of loss function with epoch')

        fig.tight_layout()  
        plt.savefig(path_figure)
        plt.close()


    def plot_gan_history_component(self, data_dict, path_figure):
        epochs = range(len(data_dict['content_loss']))
        
        plt.figure(figsize=(15, 10))

        # Plot 1: Generator Components
        plt.subplot(2, 1, 1)
        plt.plot(epochs, data_dict['content_loss'], label='Content Loss (MSE)')
        plt.plot(epochs, data_dict['adv_loss'], label='Adversarial Loss')
        plt.plot(epochs, data_dict['gen_loss'], label='Generator Loss')
        plt.title('Generator Loss Components')
        plt.legend()

        # Plot 2: Discriminator Components
        plt.subplot(2, 1, 2)
        plt.plot(epochs, data_dict['real_loss'], label='Real Loss')
        plt.plot(epochs, data_dict['fake_loss'], label='Fake Loss')
        plt.plot(epochs, data_dict['disc_loss'], label='Discriminator Loss')
        plt.title('Discriminator Loss Components')
        plt.legend()

        plt.tight_layout()
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



