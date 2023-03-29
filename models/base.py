class BaseModel():
    '''
    Constructor. This function initializes the model and trained parameters if a path is provided.
    A name and short description of the model must be provided.

    model_path: path of trained model parameters to be loaded (optional)
    returns: no output is expected
    '''
    def __init__(self, model_path = None, suffix = ''):
        self.name = ''
        self.description = ''
        pass

    '''
    This function loads trained parameters. Append /self.name + suffix to the model_path.

    model_path: path of trained model parameters to be loaded
    suffix: model path suffix
    returns: no output is expected
    '''
    def load(self, model_path, suffix = ''):
        pass

    '''
    This functions saves to model to disc. Append /self.name + suffix to the model_path.

    model_path: path that the model will be saved to
    suffix: model path suffix
    returns: no output is expected
    '''
    def save(self, model_path, suffix = ''):
        pass

    '''
    This function is used to predict a given set of texts.

    df: dataframe of texts to be predicted
    returns: list of float values of length "len(df.index)"
    '''
    def predict(self, df):
        pass

    '''
    This function is used to train the model.

    df_train: dataframe of texts to be used for training
    df_eval: dataframe of texts to be optionally used for evaluation
    returns: no output is expected
    '''
    def train(self, df_train, df_eval, labels_train, labels_eval):
        pass
