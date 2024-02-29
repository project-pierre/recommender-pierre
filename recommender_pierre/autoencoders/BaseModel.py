from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class BaseModel(object):

    @staticmethod
    def callbacks_list(path_model, monitor='val_loss', patience=15):
        """
        Callbacks of Train model
        """
        checkpoint = ModelCheckpoint(path_model, monitor=monitor, verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor=monitor, min_delta=0, patience=patience, verbose=0, mode='auto')

        callbacks_list = [checkpoint, early_stop]
        return callbacks_list
