class NoiseOversampler:
    def __init__(self, degree=0.001, amplify=2, cat_val_threshold=0.06):
        self.degree = degree
        self.amplify = amplify
        self.cat_val_threshold = cat_val_threshold
        np.random.seed(123)
        random.seed(123)

    def fit_resample(self, X, y):
        data_frame = pd.concat([X, y], axis=1)
        label_in_majority, label_in_minority, number_of_majority_labels, number_of_minority_labels = self.get_label_counts(data_frame)
        difference_num_majority_minority_labels = number_of_majority_labels - number_of_minority_labels
        df_minority = data_frame[data_frame.iloc[:, -1] == label_in_minority]
        df_sampled = self.generate_samples(df_minority, number_of_majority_labels, label_in_minority)
        df_sampled = df_sampled.drop_duplicates()[:difference_num_majority_minority_labels]
        average_decimal_points = self.get_average_decimal_points(X)
        df_sampled.iloc[:, :-1] = df_sampled.iloc[:, :-1].round(average_decimal_points)
        df_sampled.iloc[:, -1] = label_in_minority
        train_df = pd.concat([data_frame, df_sampled], axis=0).reset_index(drop=True)
        return train_df.iloc[:, :-1], train_df.iloc[:, -1]

    def get_label_counts(self, data_frame):
        label_in_majority = data_frame.iloc[:, -1].value_counts().idxmax()
        label_in_minority = data_frame.iloc[:, -1].value_counts().idxmin()
        number_of_majority_labels = data_frame.iloc[:, -1].value_counts().max()
        number_of_minority_labels = data_frame.iloc[:, -1].value_counts().min()
        return label_in_majority, label_in_minority, number_of_majority_labels, number_of_minority_labels

    def generate_samples(self, df_minority, number_of_majority_labels, label_in_minority):
        df_sampled = pd.DataFrame()
        number_of_minority_labels = 0
        while number_of_minority_labels <= number_of_majority_labels:
            negligible_noise = self.degree * self.amplify
            rows, cols = df_minority.shape
            noise = np.random.normal(loc=0, scale=negligible_noise, size=(rows, cols))
            for col in range(cols):
                unique_values = len(df_minority.iloc[:, col].unique())
                if unique_values / rows > self.cat_val_threshold:
                    df_minority.iloc[:, col] += noise[:, col]
            df_sampled = pd.concat([df_sampled, df_minority], axis=0).reset_index(drop=True)
            df_sampled.iloc[:, -1] = label_in_minority
            number_of_minority_labels = df_sampled[df_sampled.iloc[:, -1] == label_in_minority].shape[0]
            self.amplify += 1
        return df_sampled

    def get_average_decimal_points(self, X):
        average_decimal_points = X.apply(lambda col: col.apply(lambda val: len(str(val).split('.')[1]) if isinstance(val, float) and '.' in str(val) else 0).mean())
        average_decimal_points = average_decimal_points.round(0).astype(int)
        return average_decimal_points

    def custom_oversampler(self, X_oversampled, y_oversampled, n_samples_class1=509, n_samples_class0=1):
        train_df_oversampled = pd.concat([X_oversampled, y_oversampled], axis=1)
        df_target_1 = train_df_oversampled[train_df_oversampled.iloc[:, -1] == 1].sample(n=n_samples_class1, random_state=42)
        df_target_0 = train_df_oversampled[train_df_oversampled.iloc[:, -1] == 0].sample(n=n_samples_class0, random_state=42)
        selected_rows = pd.concat([df_target_1, df_target_0], ignore_index=True)
        selected_rows_X, selected_rows_y = self.fit_resample(selected_rows.iloc[:, :-1], selected_rows.iloc[:, -1])
        train_df_oversampled_X = pd.concat([X_oversampled, selected_rows_X], axis=0)
        train_df_oversampled_y = pd.concat([y_oversampled, selected_rows_y], axis=0)
        train_df_oversampled_X.drop_duplicates(inplace=True)
        train_df_oversampled_y.drop_duplicates(inplace=True)
        return self.fit_resample(train_df_oversampled_X.reset_index(drop=True), train_df_oversampled_y.reset_index(drop=True))