"""This module is used for training regression model with shift dataset"""
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

color = sns.color_palette()


class GBRRegression(object):
    """
    Gradient Boosting Regressor class
    """

    def __init__(self, file_name):
        # Importing the dataset
        self.selected = None
        self.dataset = pd.read_excel(file_name)
        self.new_columns, self.dataset_new = self.data_cleaning()
        self.x_values, self.y_values = self.get_x_y()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_values,
                                                                                self.y_values,
                                                                                test_size=0.2,
                                                                                random_state=0)

        self.regressor = GradientBoostingRegressor()

    def data_cleaning(self):
        """
        This function is used for data cleaning
        :return: Tuple
            Cleaned some categorical columns and new dataset
        """
        self.dataset.drop(["Plant", "Production_Id"], axis=1, inplace=True)
        threshold = self.dataset.shape[1] * .3
        self.dataset.dropna(thresh=threshold, axis=0, inplace=True)
        new_columns = self.dataset.iloc[:, 0:3]
        dataset_new = self.dataset.copy().iloc[:, 3:]

        # The data containing string values are converted to NaN,
        # and then the values that are NaN are filled in according to mean.
        mask = dataset_new.applymap(lambda x: isinstance(x, (int, float)))
        dataset_new = dataset_new.where(mask)
        dataset_new = dataset_new.where(pd.notna(dataset_new), dataset_new.mean(), axis="columns")

        # Done because the index reset was not done.
        dataset_new.reset_index(inplace=True)
        dataset_new.drop(["index"], axis=1, inplace=True)

        return new_columns, dataset_new

    def visualization_values_bar_plot(self, x_col, y_col, hue_col):
        """
        Visualization of values based considering machine and shift value
        :param x_col: string
            x-axis column name
        :param y_col: string
            y-axis column name
        :param hue_col: string
            hue column name
        :return: None
        """
        plt.figure(3)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=x_col, y=y_col, hue=hue_col, data=self.dataset)
        plt.title(f'{x_col} vs {y_col} based on {hue_col}')

    def get_x_y(self):
        """
        This function is used for detect and apply label encoder and one-hot encoder
        to some columns and define x and y columns.
        :return: Tuple
            x and y columns
        """
        work_center = self.new_columns["Work_Center"]
        shift = self.new_columns["Shift"]
        machine = self.new_columns["machine"]

        # Label Encoder
        work_center = LabelEncoder().fit_transform(work_center)
        shift = LabelEncoder().fit_transform(shift)
        machine = LabelEncoder().fit_transform(machine)

        # One-Hot Transformation and Dummy Variable Trap
        work_center_one_hot = pd.get_dummies(work_center,
                                             columns=["Work_Center"],
                                             prefix=["Work_Center"])
        shift_one_hot = pd.get_dummies(shift, columns=["Shift"], prefix=["Shift"])
        machine_one_hot = pd.get_dummies(machine, columns=["machine"], prefix=["machine"])

        y_values = pd.DataFrame(self.dataset_new.iloc[:, -1])

        x_values = pd.DataFrame(pd.concat([work_center_one_hot, shift_one_hot,
                                           machine_one_hot,
                                           self.dataset_new.iloc[:, :-1]], axis=1))

        x_values = pd.DataFrame(preprocessing.scale(x_values), columns=x_values.columns)
        y_values = pd.DataFrame(preprocessing.scale(y_values), columns=y_values.columns)

        return x_values, y_values

    def randomized_search(self):
        """
        Hyperparameter optimization with randomized search
        :return: dict
            Returns best parameters
        """
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=1, stop=1000, num=50)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(1, 9, num=5)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf}

        regressor_random = RandomizedSearchCV(estimator=self.regressor,
                                              param_distributions=random_grid,
                                              n_iter=100,
                                              cv=3,
                                              verbose=2,
                                              random_state=0,
                                              n_jobs=-1)

        regressor_random.fit(self.x_train, self.y_train)

        return regressor_random.best_params_

    def future_importance(self):
        """
        Future importance with Recursive Feature Elimination for getting best features
        :return: list
            Returns best features
        """
        # Recursive Feature Elimination
        selector = RFE(self.regressor, n_features_to_select=6)
        selector = selector.fit(self.x_train, self.y_train)

        # Rank the list of selected features
        rank_list = pd.DataFrame({"columns": pd.DataFrame(self.x_train).columns,
                                  "rank": list(selector.ranking_)})
        rank_list.sort_values("rank", inplace=True)
        selected = list(rank_list.head(6)["columns"])
        return selected

    def train_model(self):
        """
        Training regression model with Gradient Boosting Regressor
        :return: model
        Returns trained model
        """
        best_params = self.randomized_search()
        self.selected = self.future_importance()
        regress = GradientBoostingRegressor(n_estimators=best_params["n_estimators"],
                                            min_samples_split=best_params["min_samples_split"],
                                            min_samples_leaf=best_params["min_samples_leaf"],
                                            max_features=best_params["max_features"],
                                            max_depth=best_params["max_depth"])
        regress.fit(self.x_train[self.selected], self.y_train)
        return regress

    @staticmethod
    def plot_predicted(y_test, y_prediction):
        """
        Plot y_test and y_predicted graph
        :param y_test: list
            Actual y values
        :param y_prediction: list
            Predicted y values
        :return: None
        """

        plt.figure()
        plt.scatter(y_test, y_prediction, color='red')
        len_qq = np.arange(0, max(y_prediction), 0.001)
        plt.plot(len_qq, len_qq, color='blue')
        plt.title('Gradient Boosting Regressor')
        plt.xlabel('y_test')
        plt.ylabel('y_predicted')
        plt.show()


if __name__ == "__main__":
    gbr_regression = GBRRegression(file_name="shift_dataset.xlsx")
    gbr_regression.visualization_values_bar_plot(x_col='Machine',
                                                 y_col='Result',
                                                 hue_col='Shift')

    regressor = gbr_regression.train_model()
    y_predict = regressor.predict(gbr_regression.x_test[gbr_regression.selected])

    r_square = r2_score(gbr_regression.y_test, y_predict)
    print('r_sqr=', r_square)

    adjusted_r_squared = 1 - (1 - r_square) * (len(gbr_regression.y_values) - 1) / (
                len(gbr_regression.y_values) - gbr_regression.x_values.shape[1] - 1)
    print('adjusted_r_square=', adjusted_r_squared)

    root_mse = sqrt(mean_squared_error(gbr_regression.y_test, y_predict))
    print('root_mse=', root_mse)
