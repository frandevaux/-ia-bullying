import matplotlib.pyplot as plt
import numpy as np

# Datos
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
rf_train_results = [0.67, 0.61, 0.46, 0.53]
svm_train_results = [0.6727, 0.6344, 0.4209, 0.5061]
rf_test_results = [0.6710, 0.62, 0.45, 0.52]
svm_test_results = [0.6687, 0.6256, 0.4145, 0.4986]

# Posiciones de las barras
bar_positions_rf = np.arange(len(metrics))
bar_positions_svm = [pos + 0.2 for pos in bar_positions_rf]
bar_positions_test_rf = [pos + 0.2 for pos in bar_positions_svm]
bar_positions_test_svm = [pos + 0.2 for pos in bar_positions_test_rf]

# Tamaño de las barras
bar_width = 0.2

# Crear el gráfico de barras
plt.bar(bar_positions_rf, rf_train_results, width=bar_width, label='Random Forest Train', color='#0e9920')
plt.bar(bar_positions_svm, svm_train_results, width=bar_width, label='Support Vector Machine Train', color='#bfb80d')
plt.bar(bar_positions_test_rf, rf_test_results, width=bar_width, label='Random Forest Test', color='#49df5d')
plt.bar(bar_positions_test_svm, svm_test_results, width=bar_width, label='Support Vector Machine Test', color='#f1eb32')

# Configurar el gráfico
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Comparison of Model Performance Metrics')
plt.xticks([pos + bar_width*1.5 for pos in bar_positions_rf], metrics)
plt.legend()
plt.ylim(0, 1)

plt.savefig('./results/plots/Model_Metrics_Comparison.png')
