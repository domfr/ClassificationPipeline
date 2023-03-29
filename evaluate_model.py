import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, confusion_matrix
from models.BERT import BERTExample

from data_processing.preprocessing import split_data, extract_df
import copy


def write_output(model, i, labels, prediction, save_output, suffix):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, [pred > 0 for pred in prediction], average='binary')
    lrap = average_precision_score(labels, prediction)
    cm = confusion_matrix(labels, [pred > 0 for pred in prediction])
    print("###################### metrics ########################")
    print('LRAP: ' + str(lrap))
    print('F1: ' + str(f1))

    if save_output:

        print("saving to", 'results/' + model.name + '/predictions_' + suffix + '.txt')
        Path('results/' + model.name).mkdir(parents=True, exist_ok=True)
        with open('results/' + model.name + '/metrics_' + suffix + '.txt', 'w' if i is not None and i == 0 else 'a') as f:

            if i is None:
                f.write('Total:\n')
            else:
                f.write('Fold: ' + str(i) + '\n')
            f.write('LRAP: ' + str(lrap) + '\n')
            f.write('F1: ' + str(f1) + '\n')
            f.write('Precision: ' + str(precision) + '\n')
            f.write('Recall: ' + str(recall) + '\n')
            f.write(str(cm) + '\n')
            f.write('-----\n')


def evaluate_func(model_template=None, n_splits=5, save_output=True, model_args={}):
    if model_template is None:
        model_template = BERTExample()

    df, labels, suffix = extract_df()
    df = df.reset_index()

    predictions = np.zeros(shape=(len(labels)))
    predictions_folds = []

    print("Beginning Evaluation")
    for i, (train_index, eval_index) in enumerate(split_data("./", labels=labels, n_splits=n_splits)):
        # train_index = train_index[:1]
        # eval_index = eval_index[:1]
        print("Model copy")
        model = copy.deepcopy(model_template)
        #    model.load_encoder_model(i)
        print("training fold", i)
        print("df dim", df.shape)
        print("training examples", len(train_index))
        print("eval examples", len(eval_index))

        model.train(df.iloc[train_index], df.iloc[eval_index], labels[train_index], labels[eval_index])
        print("Saving")
        model.save(model_path='trained_models', suffix='_' + str(i))

        print("Prediction")
        prediction = model.predict(df.iloc[eval_index])

        predictions_folds.append(prediction)
        predictions[eval_index] = prediction
        write_output(model, i, labels[eval_index], prediction, save_output, suffix)

    write_output(model, None, labels, predictions, save_output, suffix)
    print("saving to", 'results/' + model.name + '/predictions_' + suffix + '.pkl')
    with open('results/' + model.name + '/predictions_' + suffix + '.pkl', 'wb') as f:
        pickle.dump(predictions_folds, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    evaluate_func(PatentBertDoubleSummary())
