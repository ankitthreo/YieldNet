import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

def plot(data, split):
    data_path = f'{data}/lambda0.1/cv{split}_ckpts_ds2s_'
    file_path = 'fold_0/interpret.csv'
    hard_perm = f'./perm_results/gold_perm/{data_path}h20/{file_path}'
    soft_perm = f'./perm_results/gold_perm/{data_path}h20_hard/{file_path}' 
    
    df_hard = pd.read_csv(hard_perm)
    df_soft = pd.read_csv(soft_perm)
    
    delta_PE = df_soft['PE'] - df_hard['PE']
    delta_AE = df_soft['AE'] - df_hard['AE']
    # t_trace = df_hard['hard_traces']
    h_trace = df_soft['hard_traces'] 
    s_trace = df_soft['soft_traces']
    # print(f"{data}_CV{split}")
    # print("Correlation b/w AE and PE: ", spearmanr(delta_PE, delta_AE))
    # print("Correlation b/w hard_traces and AE: ", spearmanr(h_trace, delta_AE))
    # print("Correlation b/w soft_traces and AE: ", spearmanr(s_trace, delta_AE))
    # print("\n")
    
    #plt.scatter(X, Y)
    #plt.savefig(f'./plots/{data}_cv{split}_plot.png')
    return h_trace, delta_PE, np.abs(delta_AE)

if __name__ == '__main__':
     fig, axes = plt.subplots(1, 3, figsize=(12, 4))
     
     for d, data in enumerate(['gpy0', 'gpy1', 'gpy2']):
         Xs = []
         Ys = []
         for i in range(1, 11):
             _, X, Y = plot(data, i)
             Xs.extend(list(X))
             Ys.extend(list(Y))
             #print("Splitwise Correlation b/w hard_traces and AE: ", spearmanr(X, Y))
             #print("\n")
         print("Datasetwise Correlation b/w hard_traces and AE: ", spearmanr(Xs, Ys))
         print("\n")
         
         max_y = max(Ys)
         Ys = [y/max_y for y in Ys]
         # Create a dictionary to store the total income and count for each age
         X_Y_dict = {}
         X_count_dict = {}
        
         # Calculate total income and count for each age
         for x, y in zip(Xs, Ys):
             if x in X_Y_dict:
                 X_Y_dict[x] += y
                 X_count_dict[x] += 1
             else:
                 X_Y_dict[x] = y
                 X_count_dict[x] = 1
         
         # Calculate average income for each age
         unique_Xs = sorted(list(set(Xs)))
         average_Ys = [X_Y_dict[x] / X_count_dict[x] for x in unique_Xs]
        
         # Create a scatter plot
         axes[d].scatter(unique_Xs, average_Ys, label='Average $\Delta$AE')
        
         # Add labels and title
         axes[d].set_xlabel('Error in $P$')
         axes[d].set_ylabel('$\Delta$AE')
         axes[d].set_title('GP'+str(d))
         #axes[d].set_title('Subplot 3')
         # Calculate R-squared
         r2_value = r2_score(unique_Xs, average_Ys)
        
         # Display R-squared on the plot
         # plt.text(0.7, 0.1, f'R-squared: {r2_value:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
         # Show legend
         axes[d].legend()
        
         # Display the plot
         # plt.show()
         '''
         plt.scatter(Xs, Ys)
         '''
         #calculate equation for trendline
         z = np.polyfit(unique_Xs, average_Ys, 1)
         p = np.poly1d(z)
        
         #add trendline to plot
         axes[d].plot(unique_Xs, p(unique_Xs), color="lightgreen")
         #axes[i].savefig(f'./plots/{data}_h300_plot.png')
         #plt.close()
         
     for ax in axes:
         ax.legend()
         ax.grid(True)
        
     plt.tight_layout()
     plt.savefig(f'./plots/gpy_h20_plot.png')
     plt.close()
     
     
         