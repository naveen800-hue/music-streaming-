# Define Metric Calculation Function
def calculateMetrics(algorithm, testY, predict):
    acc = accuracy_score(testY, predict)
    prec = precision_score(testY, predict, average='binary', pos_label=1)
    rec = recall_score(testY, predict, average='binary', pos_label=1)
    f1 = f1_score(testY, predict, average='binary', pos_label=1)
    
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    fscore.append(f1)
    
    print(f"\n📊 {algorithm} Evaluation Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(testY, predict)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{algorithm} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# --------------------------
# 🚀 Model 1: Decision Tree
# --------------------------
dt_model_path = 'DecisionTree_model.pkl'
if os.path.exists(dt_model_path):
    dt = joblib.load(dt_model_path)
    print("✅ Decision Tree model loaded.")
else:
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    joblib.dump(dt, dt_model_path)
    print("✅ Decision Tree model trained and saved.")

dt_preds = dt.predict(X_test)
calculateMetrics("Decision Tree", y_test, dt_preds)

# ----------------------------
# 🚀 Model 2: Random Forest
# ----------------------------
rf_model_path = 'RandomForest_model.pkl'
if os.path.exists(rf_model_path):
    rf = joblib.load(rf_model_path)
    print("✅ Random Forest model loaded.")
else:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, rf_model_path)
    print("✅ Random Forest model trained and saved.")

rf_preds = rf.predict(X_test)
calculateMetrics("Random Forest", y_test, rf_preds)
