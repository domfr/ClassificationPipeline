import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from data_processing.preprocessing import split_data, extract_df
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, confusion_matrix



import sys

def create_ranked_hits_fractionplot(y_true, y_pred, resolution, model, title_string, fig=True, save=True):
    # combine list of true labels and predictions into data frame
    df_ranked_preds = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    # sort data frame by descending model score
    df_ranked_preds = df_ranked_preds.sort_values(by = ['y_pred'], ascending = False)
    # add ranking column
    df_ranked_preds['rank'] = np.linspace(start = 1, stop = df_ranked_preds.shape[0], num = df_ranked_preds.shape[0])
    # recast true labels as integer for visualization purposes
    df_ranked_preds['y_true_int'] = df_ranked_preds['y_true'].astype(int)

    # create percentiles
    df_ranked_preds['y_pred_pct'] = pd.qcut(df_ranked_preds['y_pred'], q = resolution)
    # group by percentiles
    df_grp = df_ranked_preds.groupby('y_pred_pct')['y_true'].agg(['sum', 'count'])
    df_grp = df_grp.sort_index(ascending = False)
    df_grp.head()

    # build relevant measures
    df_grp['y_correct_pct'] = df_grp['sum'] / df_grp['count'] * 100
    pct_relevant = 100 * sum(df_ranked_preds['y_true_int']) / df_ranked_preds.shape[0]

    # visualize
    if fig:
        fig = plt.figure(figsize = (20, 10))    
        plt.bar(list(range(1, len(df_grp['y_correct_pct'].values) + 1))[::-1], df_grp['y_correct_pct'].values, width = 1.0, edgecolor = 'blue')  # , linewidth = 3)
        plt.gca().invert_xaxis()
        plt.title('Dashed red line: Theoretical optimum \n Blue solid line: '+model+' ranking')
        plt.title(title_string, fontsize = 24, y = 1.05)
        plt.ylabel('Relevant documents [%]')
        plt.xlabel('Percentile of ranked observations\n (highest score left, lowest scores right)')
        plt.ylim(-1.05, 101.05)
        plt.axes().grid()
    else:

        fig = plt.figure(figsize = (16,7.5))
        ax = fig.add_subplot(1, 1, 1)
        
        fig.set_figheight(8) # figure height in inches
        fig.tight_layout(pad=3.0)
        ax.bar(list(range(1, len(df_grp['y_correct_pct'].values)+1))[::-1], df_grp['y_correct_pct'].values, width = 1.0, edgecolor='blue')#, linewidth = 3)
        fig.gca().invert_xaxis()
        fig.suptitle(title_string, fontsize=24, y=1.05)
        ax.set_title('Dashed red line: Theoretical optimum \n Blue solid line: '+model+'  ranking', y=1, pad=10)
        ax.set_ylabel('Relevants documents [%]')
        ax.set_xlabel('Percentile of ranked observations\n (highest score left, lowest scores right)')  
        ax.set_ylim(-1.05,101.05)
        ax.grid()
    point1 = [100, 100]
    point2 = [100 - pct_relevant, 100]
    point3 = [100 - pct_relevant, 0]
    point4 = [0, 0]
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    if fig:
        plt.plot(x_values, y_values, color = "red", linestyle = "--")
    else:
        ax.plot(x_values, y_values, color = "red", linestyle="--")
    x_values = [point2[0], point3[0]]
    y_values = [point2[1], point3[1]]
    if fig:
        plt.plot(x_values, y_values, color = "red", linestyle = "--")
    else:
        ax.plot(x_values, y_values, color = "red", linestyle="--")
    x_values = [point3[0], point4[0]]
    y_values = [point3[1], point4[1]]
    if fig:
        plt.plot(x_values, y_values, color = "red", linestyle = "--")
        
    else:
        ax.plot(x_values, y_values, color = "red", linestyle="--")

    if save:
        plt.savefig('../results/'+model+'/ranked_hits.png')
        plt.show()



def load_results(model, suffix, model_path = "./results/",data_path= "", split_path="./", n_splits=5):
    predictions = pickle.load(open(model_path + model + "/predictions"+suffix+".pkl", "rb"))
    y_true = []
    y_pred = []
    df, labels, suffix = extract_df( path = data_path)
    for i, (train_index, eval_index) in enumerate(split_data(split_path,  n_splits = n_splits)):
        y_true.extend(labels[eval_index])
        y_pred.extend(predictions[i])
        break

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, [pred > 0 for pred in y_pred], average = 'binary')
    lrap = average_precision_score(y_true, y_pred)
    cm = confusion_matrix(y_true, [pred > 0 for pred in y_pred])
    print("###################### metrics ########################")
    print('LRAP: ' + str(lrap))
    print('F1: ' + str(f1))
    print('cm: ' + str(cm))
    print('recall: ' + str(recall))
    print('precision: ' + str(precision))

    return y_true,y_pred


def load_results_folds(model, suffix, model_path = "./results/",data_path= "", split_path="./", n_splits=5):
    predictions = pickle.load(open(model_path + model + "/predictions"+suffix+".pkl", "rb"))
    y_true = []
    y_pred = []
    df, labels, suffix = extract_df( path = data_path)
    for i, (train_index, eval_index) in enumerate(split_data(split_path,  n_splits = n_splits)):
        y_true.append(labels[eval_index])
        y_pred.append(predictions[i])

    return y_true,y_pred

def plot_model_results(model, suffix, model_path = "./results/" , data_path = "",  split_path="./", fig=True):
    y_true,y_pred = load_results(model, suffix, model_path= model_path, data_path = data_path,  split_path=  split_path)
    create_ranked_hits_fractionplot(y_true, y_pred, 100, model, 'Ranking of relevant documents: Model '+model, fig=fig)
    

if __name__ == "__main__":
   # model = sys.argv[1]
   # #  if model=="":
   #     model="BERT_Example"
    plot_model_results("", "")
