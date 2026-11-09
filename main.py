import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))


def main():

    print("--- 1. Data Preparation ---")
    df = pd.read_csv("Diamonds_4000.csv")

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    print(df.describe())
    print(df.info())

    df = df.dropna()
    df = df.drop_duplicates()
    # df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]

    le_color = LabelEncoder()
    le_clarity = LabelEncoder()
    df['color'] = le_color.fit_transform(df['color'])
    df['clarity'] = le_clarity.fit_transform(df['clarity'])

    print(f"\nFinal number of records after cleaning: {len(df)}")
    print("Classes distribution ('cut'):\n", df['cut'].value_counts())



    print("\n--- 2. Visualizing scatter plots ---")
    features_for_plot = ['carat', 'depth', 'table', 'price', 'cut']
    sns.pairplot(df[features_for_plot], hue='cut', palette='viridis', corner=True)
    plt.suptitle("Scatter Plot Matrix", y=1.02)
    plt.show()


    X = df.drop(columns=['cut'])
    y = df['cut']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    print("\n--- 4-6. Base KNN (Without Scaling) ---")
    # knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix (Without Scaling)")
    plt.show()


    print("\n--- 7. KNN (With Feature Scaling) ---")
    # scaler = RobustScaler()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_scaled = KNeighborsClassifier(n_neighbors=5)
    knn_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = knn_scaled.predict(X_test_scaled)

    print(classification_report(y_test, y_pred_scaled))




    print("\n--- 8. Finding best parameters ---")
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 15]}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")



    print("\n--- 10. Custom Distance Check ---")
    X_test_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X.columns)
    X_train_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X.columns)

    sample_indices = y_test.groupby(y_test).head(3).index

    for idx in sample_indices:
        test_instance = X_test_df.loc[idx].values
        true_class = y_test.loc[idx]

        min_dist = float('inf')
        nearest_train_idx = None

        for train_idx in X_train_df.index:
            train_instance = X_train_df.loc[train_idx].values
            dist = euclidean_distance(test_instance, train_instance)

            if dist < min_dist:
                min_dist = dist
                nearest_train_idx = train_idx

        nearest_class = y_train.loc[nearest_train_idx]
        match_status = "MATCH" if true_class == nearest_class else "MISMATCH"

        print(
            f"Test ID: {idx} (Real: {true_class[:4]}) -> NN Train ID: {nearest_train_idx} (Pred: {nearest_class[:4]}) | {match_status}")


if __name__ == "__main__":
    main()