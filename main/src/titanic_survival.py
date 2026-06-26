import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')



BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'titanic.csv')
PLOTS_DIR = os.path.join(BASE_DIR, '..', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 55)
print("  TITANIC SURVIVAL PREDICTION")
print("=" * 55)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nSurvival rate: {df['Survived'].mean()*100:.1f}% survived\n")


print("[2] EDA Summary")
print(f"    Missing - Age: {df['Age'].isnull().sum()} | "
    f"Cabin: {df['Cabin'].isnull().sum()} | "
    f"Embarked: {df['Embarked'].isnull().sum()}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Titanic - Exploratory Data Analysis', fontsize=16, fontweight='bold')

sns.countplot(x='Survived', data=df, palette=['#e74c3c', '#2ecc71'], ax=axes[0, 0])
axes[0, 0].set_title('Survival Count')

axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(['Died (0)', 'Survived (1)'])
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

sns.countplot(x='Sex', hue='Survived', data=df,
            palette=['#e74c3c', '#2ecc71'], ax=axes[0, 1])
axes[0, 1].set_title('Survival by Gender')
axes[0, 1].legend(['Died', 'Survived'])


sns.countplot(x='Pclass', hue='Survived', data=df,
            palette=['#e74c3c', '#2ecc71'], ax=axes[0, 2])
axes[0, 2].set_title('Survival by Ticket Class')
axes[0, 2].legend(['Died', 'Survived'])


df['Age'].dropna().plot(kind='hist', bins=30, color='#3498db',
                        edgecolor='white', ax=axes[1, 0])
axes[1, 0].set_title('Age Distribution')
axes[1, 0].set_xlabel('Age')

df[
'Fare'].plot(kind='hist', bins=40, color='#9b59b6',
                edgecolor='white', ax=axes[1, 1])
axes[1, 1].set_title('Fare Distribution')
axes[1, 1].set_xlabel('Fare')


survival_pivot = df.pivot_table('Survived', index='Pclass', columns='Sex')
survival_pivot.plot(kind='bar', ax=axes[1, 2], color=['#e74c3c', '#3498db'],
                    edgecolor='white')
axes[1, 2].set_title('Survival Rate by Class & Gender')
axes[1, 2].set_xlabel('Ticket Class')
axes[1, 2].set_ylabel('Survival Rate')
axes[1, 2].set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
axes[1, 2].legend(['Female', 'Male'])

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_plots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    EDA plots saved.\n")



print("[3] Feature Engineering")
data = df.copy()

data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'
)
data['Title'] = data['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
print(f"    Titles found: {data['Title'].value_counts().to_dict()}")

data['Age'] = data.groupby('Title')['Age'].transform(
    lambda x: x.fillna(x.median())

)



data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)


data['FareBand'] = pd.qcut(data['Fare'], 4, labels=False, duplicates='drop')

data['AgeBand'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior'])


data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

data['Cabin'] = data['Cabin'].fillna('U')
data['Deck'] = data['Cabin'].str[0]

print(f"    FamilySize range: 1-{data['FamilySize'].max()}")
print(f"    Passengers travelling alone: {data['IsAlone'].sum()}\n")

print("[4] Encoding & Feature Selection")

for col in ['Sex', 'Title', 'AgeBand', 'Embarked', 'Deck']:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
            'Fare', 'Embarked', 'Title', 'FamilySize',
             'IsAlone', 'FareBand', 'AgeBand', 'Deck']


X = data[FEATURES]
y = data['Survived']
print(f"    Features used: {FEATURES}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[5] Split -> Train: {len(X_train)} | Test: {len(X_test)}\n")

print("[6] Training Models")
print("-" * 45)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=6,
                                                random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200,
                                                    learning_rate=0.05,
                                                    max_depth=4,
                                                    random_state=42),

}



results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    cv  = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()

    results[name] = {'model': model, 'y_pred': y_pred,
                    'y_proba': y_proba, 'acc': acc,
                    'roc': roc, 'cv': cv}

    print(f"  {name}")
    print(f"    Accuracy : {acc*100:.2f}%")
    print(f"    ROC-AUC  : {roc:.4f}")
    print(f"    CV Score : {cv*100:.2f}%")
    print()


best_name = max(results, key=lambda k: results[k]['roc'])
best      = results[best_name]
print(f"[7] Best Model: {best_name}")
print("-" * 45)
print(classification_report(y_test, best['y_pred'],
                        target_names=['Died', 'Survived']))


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Model Evaluation - {best_name}', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'], ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')


for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['roc']:.3f})")
axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1].set_title('ROC Curve - All Models')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=8)


if hasattr(best['model'], 'feature_importances_'):
    importance_model      = best['model']
    importance_model_name = best_name
else:
    importance_model      = results['Random Forest']['model']
    importance_model_name = 'Random Forest'


importances = pd.Series(importance_model.feature_importances_, index=FEATURES)
importances.sort_values().plot(kind='barh', color='#3498db',
                            edgecolor='white', ax=axes[2])
axes[2].set_title(f'Feature Importance ({importance_model_name})')
axes[2].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_evaluation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n    Evaluation plots saved.\n")

print("[8] Sample Predictions (first 10 test passengers)")
print("-" * 45)

sample = X_test.iloc[:10].copy()
sample['Actual']    = y_test.iloc[:10].values
sample['Predicted'] = best['model'].predict(sample[FEATURES])
sample['Correct']   = sample['Actual'].values == sample['Predicted']
print(sample[['Pclass', 'Sex', 'Age', 'Actual', 'Predicted', 'Correct']].to_string())


print("\n" + "=" * 55)
print("  DONE — All plots saved to plots/")
print("=" * 55)
