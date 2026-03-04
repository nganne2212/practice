from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

print("=" * 50)
print("HUẤN LUYỆN MÔ HÌNH IRIS CLASSIFIER")
print("=" * 50)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nSố lượng mẫu huấn luyện: {X_train.shape[0]}")
print(f"Số lượng mẫu kiểm tra: {X_test.shape[0]}")
print(f"Số lượng đặc tính: {X_train.shape[1]}")
print(f"Số lượng lớp: {len(np.unique(y))}")

# Huấn luyện mô hình Random Forest (tốt hơn Logistic Regression)
print("\n🔄 Đang huấn luyện mô hình Random Forest...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# Đánh giá
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

print(f"\n✅ Độ chính xác trên tập huấn luyện: {train_score*100:.2f}%")
print(f"✅ Độ chính xác trên tập kiểm tra: {test_score*100:.2f}%")

# Cross-validation
cv_scores = cross_val_score(clf, X_scaled, y, cv=5)
print(f"✅ Cross-validation score (trung bình): {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# Đặc tính quan trọng
feature_importance = clf.feature_importances_
feature_names = iris.feature_names
print("\n📊 Mức độ quan trọng của từng đặc tính:")
for name, importance in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {importance*100:.2f}%")

# Lưu mô hình và scaler
print("\n💾 Đang lưu mô hình...")
pickle.dump(clf, open("iris_model.pkl", 'wb'))
pickle.dump(scaler, open("iris_scaler.pkl", 'wb'))
print("✅ Lưu thành công!")
print("=" * 50)
