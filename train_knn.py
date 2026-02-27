
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sys


def main():
    # Load dataset
    df = pd.read_csv('cancer patient data sets.csv')

    # Identify target column
    possible_targets = ['diagnosis', 'target', 'label', 'Outcome', 'Class', 'status']
    target = None
    for t in possible_targets:
        if t in df.columns:
            target = t
            break
    if target is None:
        target = df.columns[-1]

    # Drop common ID columns if present
    for idc in ['id', 'ID', 'patient_id', 'PatientID']:
        if idc in df.columns and idc != target:
            df = df.drop(columns=[idc])

    df = df.dropna(how='all')

    # Separate features and target
    # Convert non-numeric features (except target) to numeric codes
    feature_cols = [c for c in df.columns if c != target]
    for col in feature_cols:
        if df[col].dtype == object or str(df[col].dtype).startswith('category'):
            df[col] = df[col].astype('category').cat.codes

    # Encode target if needed
    le = None
    if df[target].dtype == object or str(df[target].dtype).startswith('category'):
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target].astype(str))

    # Fill numeric missing values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    X = df.drop(columns=[target])
    y = df[target]

    # Train/test split (use stratify when possible)
    try:
        if len(np.unique(y)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Grid search for best KNN params
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance']
    }
    knn = KNeighborsClassifier()
    gs = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)

    print('Best params:', gs.best_params_)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification report:\n', classification_report(y_test, y_pred))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

    # Save model, scaler, and encoder
    joblib.dump({'model': best, 'scaler': scaler, 'label_encoder': le}, 'knn_model.joblib')

    # Save predictions
    pd.DataFrame({'y_true': y_test.values, 'y_pred': y_pred}).to_csv('knn_predictions.csv', index=False)


if __name__ == '__main__':
    main()
