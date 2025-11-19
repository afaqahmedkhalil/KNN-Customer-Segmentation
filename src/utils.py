import matplotlib.pyplot as plt


def plot_k_selection(k_range, k_scores):
    plt.plot(k_range, k_scores, marker='o')
    plt.xlabel('K value')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Finding the Best K')
    plt.grid(True)
    plt.show()
