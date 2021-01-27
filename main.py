from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
breast_cancer_data = load_breast_cancer()

# View dataset
# print(breast_cancer_data.data[0])
# print(breast_cancer_data.feature_names)

# View labels
# print(breast_cancer_data.target[0])
# print(breast_cancer_data.target_names)

# Split dataset
training_data, validation_data, training_labels, validation_labels = train_test_split(
    breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=1)

# Build model
k_list = range(1, 100)
accuracies = []
for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    score = classifier.score(validation_data, validation_labels)
    accuracies.append(score)

# Plot settings
sns.set_style("darkgrid")
sns.set_context("notebook")

# Plot
sns.lineplot(x=k_list, y=accuracies)
plt.xlabel("k")
plt.ylabel("Validation accuracy")
plt.title("Breast cancer classifier accuracy")
plt.show()
