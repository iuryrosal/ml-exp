### ** Aplicar Experimentação Contínua na Avaliação de Modelos**

### **1. Definir o Cenário de Comparação**

- **Hipótese nula (H0H_0H0)**: Não há diferença significativa entre as performances dos modelos.
- **Hipótese alternativa (H1H_1H1)**: Pelo menos um modelo tem uma performance significativamente diferente.

### **2. Escolher a Métrica de Avaliação**

Utilize métricas relevantes para o problema, como:

- Regressão: RMSE, R2, MAE, R2R^2
- Classificação: AUC-ROC, F1-score, precisão, recall.

### **3. Coletar Resultados de Teste**

- Divida os dados em múltiplos folds (ex.: cross-validation) ou utilize diferentes conjuntos de teste.
- Para cada fold ou conjunto de teste, registre a métrica de avaliação para cada modelo.

### **4. Selecionar o Teste Estatístico**

Baseie a escolha do teste em:

- **Normalidade dos Dados**:
    - Use o **Teste de Shapiro-Wilk** para verificar se as métricas seguem uma distribuição normal.
- **Homocedasticidade**:
    - Use os testes de **Levene** ou **Bartlett** para avaliar se as variâncias entre os modelos são iguais.

**Casos possíveis:**

- **Dados normais e homocedásticos**: Use **ANOVA**.
- **Dados não normais ou não homocedásticos**: Use **Kruskal-Wallis**.
- **Comparação entre dois modelos**:
    - **Dados normais**: Use **Teste t de Student**.
    - **Dados não normais**: Use **Mann-Whitney**.

### **5. Aplicar Testes Post-Hoc (Se Relevante)**

Se a ANOVA ou Kruskal-Wallis indicar diferenças significativas, use testes post-hoc para identificar quais modelos são significativamente diferentes:

- **ANOVA**: Use o **Teste de Tukey**.
- **Kruskal-Wallis**: Use comparações par a par com **Mann-Whitney**.