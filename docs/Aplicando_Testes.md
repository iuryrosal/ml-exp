# Cenário: Otimização de Campanhas de Marketing em um E-commerce

Suponha que uma empresa de e-commerce quer otimizar suas campanhas de marketing para aumentar a **taxa de conversão** (número de visitantes que se tornam compradores) e o **ticket médio de compra** (valor médio gasto por cliente). Eles implementam três abordagens de marketing:

1. **Campanha A**: Descontos em produtos populares.
2. **Campanha B**: Programa de pontos para fidelização.
3. **Campanha C**: Recomendações personalizadas com base no comportamento do usuário.

Para avaliar o impacto dessas campanhas, a empresa coleta dados de conversão e ticket médio de três grupos de clientes, cada um exposto a uma das campanhas. O objetivo é identificar qual campanha é mais eficaz, utilizando métodos estatísticos rigorosos para validar os resultados.

### Etapa 1: Definição das Hipóteses

- **Hipótese Nula (H_0)**: Não há diferença significativa entre as campanhas em termos de taxa de conversão e ticket médio.
- **Hipótese Alternativa (H1H_1H1)**: Pelo menos uma das campanhas é significativamente diferente em relação às outras em termos de taxa de conversão e/ou ticket médio.

### Métricas de Avaliação

1. **Taxa de Conversão**: Proporção de usuários que realizaram uma compra após ver a campanha.
2. **Ticket Médio**: Valor médio gasto por cliente.

Essas métricas serão comparadas entre os grupos de campanhas usando diferentes testes estatísticos.

### Aplicação dos Testes Estatísticos

### 1. **Testes de Normalidade**: Verificar a Distribuição dos Dados

Antes de comparar os grupos, é importante verificar se as métricas (taxa de conversão e ticket médio) seguem uma **distribuição normal**. Isso é crucial para decidir se testes paramétricos (como ANOVA e teste t) são apropriados.

- **Teste de Shapiro-Wilk** e **Anderson-Darling** podem ser aplicados individualmente a cada métrica de cada grupo (Campanha A, B e C) para verificar a normalidade dos dados.
- **Kolmogorov-Smirnov** também pode ser utilizado como um teste geral para normalidade.

### 2. **Testes de Homocedasticidade**: Verificar Igualdade das Variâncias

Se os dados forem normalmente distribuídos, o próximo passo é verificar a **homogeneidade das variâncias** entre os grupos. Isso é necessário para determinar se podemos aplicar a ANOVA com segurança.

- **Teste de Levene** ou **Teste de Bartlett** podem ser aplicados para verificar se as variâncias de taxa de conversão e ticket médio são homogêneas entre os grupos (Campanha A, B e C).
- **Brown-Forsythe** pode ser uma alternativa menos sensível para dados que podem ter leves desvios da normalidade.

### 3. **ANOVA** e **Teste de Tukey**: Comparação entre as Campanhas

Assumindo que as variâncias são homogêneas e que os dados seguem uma distribuição normal, podemos aplicar a **ANOVA** para testar se existe uma diferença significativa na taxa de conversão e no ticket médio entre as campanhas.

- Se a ANOVA indicar uma diferença significativa, aplicamos o **Teste de Tukey** como uma análise pós-hoc para identificar quais pares de campanhas têm diferenças estatisticamente significativas.

### 4. **Testes Não Paramétricos**: Alternativas à ANOVA para Dados Não Normais

Se os testes de normalidade indicarem que os dados não seguem uma distribuição normal, ou se os testes de homocedasticidade falharem, usamos testes não paramétricos:

- **Teste de Kruskal-Wallis**: Este teste substitui a ANOVA quando não podemos assumir normalidade. Ele compara as medianas de taxa de conversão e ticket médio entre os três grupos.
- **Teste de Mann-Whitney**: Caso seja necessário comparar apenas duas campanhas de forma independente, o teste de Mann-Whitney pode ser aplicado como uma alternativa não paramétrica ao teste t para duas amostras.
- **Teste de Wilcoxon Signed-Rank**: Se houver campanhas com clientes emparelhados (ex.: antes e depois de implementar a Campanha C em um mesmo grupo), o teste de Wilcoxon é usado para comparar as medições antes e depois.

### Exemplo de Interpretação dos Resultados

1. **Normalidade e Homocedasticidade**: Suponha que os testes de Shapiro-Wilk e Bartlett indiquem que os dados de taxa de conversão seguem uma distribuição normal e têm variâncias homogêneas entre os grupos, permitindo o uso de ANOVA.
2. **Resultado da ANOVA**: A ANOVA revela um p-valor abaixo de 0,05 para taxa de conversão, indicando que há uma diferença significativa entre pelo menos duas campanhas.
3. **Teste de Tukey**: O teste de Tukey é realizado como análise pós-hoc e mostra que a Campanha C (recomendações personalizadas) tem uma taxa de conversão significativamente maior que as campanhas A e B.
4. **Testes Não Paramétricos**: Suponha que o ticket médio não segue uma distribuição normal. O teste de Kruskal-Wallis é então aplicado e revela uma diferença significativa entre as campanhas para o ticket médio. Em seguida, podemos usar Mann-Whitney para comparar pares específicos e descobrir que o ticket médio é maior na Campanha C do que nas outras.

### Conclusão da Análise

Após aplicar todos esses testes, a empresa de e-commerce conclui que:

- A **Campanha C** (recomendações personalizadas) aumenta significativamente tanto a **taxa de conversão** quanto o **ticket médio** em comparação com as outras campanhas.
- Os testes estatísticos confirmam que as diferenças observadas são estatisticamente significativas, dando uma base sólida para que a empresa priorize a Campanha C em sua estratégia de marketing.