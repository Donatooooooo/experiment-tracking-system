from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras_tuner import RandomSearch
from keras.src.optimizers import Adam
from models.model import Model
import sys, numpy as np
from os import path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PATH = 'Utils/best_params/checkpoints/mlp.weights.h5'


class MLPTrainer(Model):
    def __init__(self, target_column, drop_columns, dataset):
        self.dataset = dataset
        self.target_column = target_column
        self.drop_columns = drop_columns
        self.X_train, self.X_test, self.y_train, self.y_test = self.loadData()

    def loadData(self):
        y = self.dataset.getColumn(self.target_column)
        self.dataset.dropDatasetColumns(self.drop_columns)
        X = self.dataset.getDataset()
        X_scaled = StandardScaler().fit_transform(X)

        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def build_model(self, hp, input_dim):
        model = Sequential(
            [
                Dense(
                    units=hp.Int("units_layer1", min_value=50, max_value=200, step=25),
                    activation=hp.Choice("activation_layer1", values=["relu", "tanh"]),
                    input_dim=input_dim,
                ),
                Dropout(
                    rate=hp.Float(
                        "dropout_layer1", min_value=0.1, max_value=0.5, step=0.1
                    )
                ),
                Dense(
                    units=hp.Int("units_layer2", min_value=25, max_value=150, step=25),
                    activation=hp.Choice("activation_layer2", values=["relu", "tanh"]),
                ),
                Dropout(
                    rate=hp.Float(
                        "dropout_layer2", min_value=0.1, max_value=0.5, step=0.1
                    )
                ),
                Dense(
                    units=hp.Int("units_layer3", min_value=10, max_value=100, step=10),
                    activation=hp.Choice("activation_layer3", values=["relu", "tanh"]),
                ),
                Dropout(
                    rate=hp.Float(
                        "dropout_layer3", min_value=0.1, max_value=0.5, step=0.1
                    )
                ),
                Dense(1),
            ]
        )
        model.compile(
            optimizer=Adam(
                learning_rate=hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4])
            ),
            loss="mean_squared_error",
        )
        return model

    def findBestParams(self, max_trials=10, executions_per_trial=2):
        input_dim = self.X_train.shape[1]
        tuner = RandomSearch(
            lambda hp: self.build_model(hp, input_dim),
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory="Utils/best_params/mlp_tuning",
            project_name="mlp_search",
        )

        tuner.search(
            self.X_train,
            self.y_train,
            validation_split=0.2,
            epochs=20,
            batch_size=32,
            verbose=1,
        )
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models(num_models=1)[0]

        self.params = {
            "units_layer1": best_hps.get("units_layer1"),
            "activation_layer1": best_hps.get("activation_layer1"),
            "dropout_layer1": best_hps.get("dropout_layer1"),
            "units_layer2": best_hps.get("units_layer2"),
            "activation_layer2": best_hps.get("activation_layer2"),
            "dropout_layer2": best_hps.get("dropout_layer2"),
            "units_layer3": best_hps.get("units_layer3"),
            "activation_layer3": best_hps.get("activation_layer3"),
            "dropout_layer3": best_hps.get("dropout_layer3"),
            "learning_rate": best_hps.get("learning_rate"),
        }
        self.model = best_model
        return None

    def run(self):
        if not path.exists(PATH):
            checkpoint_callback = ModelCheckpoint(
                filepath=PATH,
                monitor="val_loss",
                verbose=1,
                save_weights_only=True,
                save_best_only=True,
                mode="min",
            )
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
            self.model.fit(
                self.X_train,
                self.y_train,
                epochs=100,
                batch_size=10,
                validation_split=0.2,
                callbacks=[early_stopping, checkpoint_callback],
            )
        else:
            self.model.load_weights(PATH)

        self.evaluate_model(self.model, self.X_test, self.y_test)
        return None

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        self.metrics = {
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MSLE": mean_squared_log_error(y_test, y_pred)
        }

        self.X = X_test
        self.Y = y_test
        print("Model trained and evaluated\n")
        return None




