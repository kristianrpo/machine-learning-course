# Conclusiones: Análisis de Regresión Lineal y Logística

**Notebook:** Sesión 02: Regresión Lineal y Logística - Guía Completa  
**Profesor:** Marco Terán  
**Curso:** Machine Learning - EAFIT  
**Fecha:** 2025

---

## 1. Introducción

Este documento analiza específicamente los resultados de cada implementación del notebook original desarrollado por el profesor Marco Terán. El objetivo es proporcionar interpretaciones detalladas de las métricas, visualizaciones y comportamientos observados en los diferentes modelos de regresión, facilitando la comprensión profunda de los conceptos fundamentales del Machine Learning.

---

## 2. Regresión Lineal Simple

### 2.1 Implementación desde Cero

**Descripción:**

Este punto implementa manualmente el algoritmo de regresión lineal simple utilizando el **método de mínimos cuadrados ordinarios (OLS)**. La implementación calcula directamente la pendiente (θ₁) y el intercepto (θ₀) mediante las fórmulas analíticas, sin depender de librerías externas. El modelo ajusta una línea recta que minimiza la suma de los errores cuadrados entre los valores predichos y reales.

**Resultados obtenidos:**
- **Ecuación:** y = 10.43 + 2.41x
- **R² = 0.9403**
- **MSE = 3.2263**
- **MAE = 1.4021**

**Interpretación Detallada:**

1. **Coeficientes del modelo:**
   - **θ₀ = 10.4302 (Intercepto):** Representa el valor base de y cuando x=0. En el contexto de datos sintéticos generados con y = 2.5x + 10 + ruido, nuestro modelo recuperó el intercepto real (10.00) con una desviación de apenas 0.43 unidades (4.3% de error aprox).
   - **θ₁ = 2.4080 (Pendiente):** Indica que por cada unidad de aumento en x, y aumenta 2.41 unidades. Comparado con el valor real (2.50), tenemos un error del 3.6%, atribuible al ruido gaussiano (σ=2) agregado intencionalmente.

2. **Métricas de error:**
   
   **MAE = 1.4021:**
   - En promedio, nuestras predicciones se desvían ±1.40 unidades del valor real.
   - Como un ejemplo práctico, si el modelo predice y=20, el valor real estará típicamente entre 18.60 y 21.40.
   - MAE Es una métrica robusta y fácil de interpretar porque está en las mismas unidades que la variable objetivo. No penaliza desproporcionadamente los errores grandes.
   
   **MSE = 3.2263:**
   - El promedio de los errores al cuadrado es 3.23 aproximadamente.
   - MSE es diferenciable en todos los puntos, lo que lo hace útil para optimización mediante gradient descent. Sin embargo, es más sensible a valores atípicos.
   
   **RMSE = √3.2263 ≈ 1.80:**
   - Desviación estándar de las predicciones de ±1.80 unidades.
   - RMSE (1.80) > MAE (1.40) confirma asimetría en la distribución de errores, lo que probablemente indica mayor penalizacion de errores grandes.
   - Al estar en las mismas unidades que y, es más interpretable que MSE pero sigue penalizando errores grandes.

3. **Coeficiente de Determinación R²:**
   
   **R² = 0.9403:**
   - El modelo explica el **94.03%** de la variabilidad total en los datos. Esto nos dice que el **el modelo lineal ajusta bien los datos**
   - El porcentaje restante **(5.7%)** Es la varianza no explicada, atribuible al:
     - Ruido aleatorio agregado (σ=2)
     - Pequeñas no-linealidades no capturadas (Recordemos que R² nos permite determinar es relaciones lineales)
   - Valida supuesto de linealidad.


4. **Análisis de residuos:**
   ![Diagram01](../_assets/image.png)

   **Gráfico izquierdo (Ajuste):**
   - Las líneas discontinuas verticales representan los **residuos individuales** (εᵢ = yᵢ - ŷᵢ).
   - Distribución uniforme arriba y abajo de la línea indica que el modelo no está sesgado sistemáticamente.
   
   **Gráfico derecho (Residuos vs Predicciones):**
   - Aproximadamente 95% de residuos caen dentro de la **Banda ±2σ**, cumpliendo con expectativa de normalidad.
   - Solo puntos aislados fuera de ±2σ, dentro de lo esperado estadísticamente.

---

### 2.2 Comparación con Scikit-learn

**Descripción:**

Este punto compara nuestra implementación manual con la implementación de scikit-learn para validar la corrección matemática. Ambos métodos deben converger a la misma solución óptima si están implementados correctamente.

**Resultados obtenidos:**

| Método              | θ₀ (Intercepto) | θ₁ (Pendiente) | R²     |
|---------------------|-----------------|----------------|--------|
| Implementación      | 10.4302         | 2.4080         | 0.9403 |
| Scikit-learn        | 10.4302         | 2.4080         | 0.9403 |
| **Diferencia**      | 0.0000          | 0.0000         | 0.0000 |

**Interpretación Detallada:**

1. **Convergencia perfecta:**
   
   - Coincidencia hasta **4 decimales** en todos los parámetros.
   - La implementación manual es equivalente a lo propuesto por la librería para 4 decimales por lo menos en la convergencia que genera.

2. **Diferencias metodológicas internas:**
   Bajo curiosidad, decidí buscar si la implementación que usamos manualmente utilizaba el mismo método que utilizó el profesor, y obtuve lo siguiente:

   **Nuestra implementación (Ecuación Normal):**
   ```
   θ₁ = Cov(X,Y) / Var(X) = Σ((xᵢ-x̄)(yᵢ-ȳ)) / Σ((xᵢ-x̄)²)
   θ₀ = ȳ - θ₁x̄
   ```
   - **Ventaja:** Solución directa, no iterativa, fácil de entender.
   - **Desventaja:** Puede ser numéricamente inestable con matrices mal condicionadas.
   
   **Scikit-learn (Descomposición SVD):**
   ```
   Usa: X = UΣVᵀ para resolver θ = (XᵀX)⁻¹Xᵀy
   ```
   - **Ventaja:** Más robusto numéricamente, maneja mejor multicolinealidad (variables independientes muy correlacionadas).
   - **Desventaja:** Ligeramente más costoso computacionalmente.

3. **¿Por qué los resultados son idénticos?**
   
   - No hay problemas numéricos (multicolinealidad, overflow, underflow).
   - El sistema tiene solución única porque X es de rango completo.
   - La función de costo es convexa (un solo mínimo).

---

### 2.3 Efecto de Outliers en Regresión Lineal

**Descripción:**

Este experimento demuestra cómo los valores atípicos (outliers) afectan dramáticamente la regresión lineal. Se comparan tres escenarios: datos sin outliers, con 1 outlier, y con 3 outliers, mostrando el impacto progresivo en el ajuste del modelo.

**Resultados observados:**

| Escenario          | R²     | Cambio vs Original | Impacto                     |
|--------------------|--------|--------------------|-----------------------------|
| Sin Outliers       | 0.975  | Baseline           | Ajuste excelente            |
| Con 1 Outlier      | 0.770  | **-21.0%**         | Degradación significativa   |
| Con 3 Outliers     | 0.682  | **-30.1%**         | Degradación severa          |

**Interpretación Detallada:**

![Impacto de Outliers](../_assets/image-01.png)

1. **Mecanismo de sensibilidad de OLS:**

    La regresión por mínimos cuadrados minimiza: **J(θ) = Σᵢ (yᵢ - ŷᵢ)²**

    **El problema:** Un outlier con error ε = 25 contribuye ε² = **625** al costo, mientras que 25 puntos normales con ε = 1 solo contribuyen 25. El modelo "sacrifica" el ajuste general para minimizar este único error grande.

   `` Por ello, se destaca la importancia de los outliers  ``

2. **Análisis visual de los escenarios:**

    **Sin Outliers (R² = 0.975):**
    - La línea pasa por el centro de masa de los datos
    - Residuos pequeños y uniformemente distribuidos
    - El modelo captura correctamente la tendencia lineal

    **Con 1 Outlier (R² = 0.770):**
    - La línea se **inclina visiblemente** hacia el outlier definido
    - Pérdida del 21% en capacidad explicativa
    - El outlier tiene **alto leverage** (lejos en X) y **alto residuo** (lejos en Y)

    **Con 3 Outliers (R² = 0.682):**
    - La línea **ya no representa** la tendencia real de los datos principales
    - Degradación del 30% en R²
    - Los outliers distorsionan completamente el modelo
    - La pendiente aumenta significativamente


---

## 3. Regresión Lineal Múltiple

### 3.1 Regresión Múltiple - Dataset California Housing

**Descripción:**

Este punto introduce la **regresión lineal múltiple**, donde el modelo aprende a predecir una variable objetivo usando múltiples características simultáneamente. Se utiliza el dataset real de California Housing, que contiene información sobre precios de viviendas y características demográficas/geográficas de diferentes distritos (el que veniamos evaluando dentro de la clase).

**Características del Dataset:**
- **20,640 muestras** (distritos censales de California)
- **8 variables predictoras:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Variable objetivo:** Precio medio de casas (en cientos de miles de dólares)

**Distribución de Variables:**

![Distribución California Housing](../_assets/image-02.png)

**Observaciones de las distribuciones y analisis descriptivo:**

1. **MedInc (Ingreso Medio):** La distribución de MedInc muestra un sesgo positivo (derecha), con la mayoría de los valores concentrados entre 1 y 5, y una cola hacia la derecha. Esto indica que la mayoría de las observaciones están en el rango inferior de ingresos, mientras que los valores más altos son relativamente raros.

2. **HouseAge (Edad de la Casa):** La variable HouseAge muestra una distribución multimodal con picos alrededor de los valores 15 y 35-40, lo que sugiere que existen ciertos rangos de edad de casas más comunes en el dataset. Además, hay una fuerte presencia de casas con edades cercanas a 50 años, lo que podría ser un límite o techo de la variable. La desviación estándar de 12.59 indica una variabilidad moderada en las edades de las casas. La mínima es 1, lo que podría reflejar construcciones recientes, y la máxima es 52, lo que sugiere una variedad de casas de diferentes épocas.

3. **AveRooms (Promedio de Habitaciones):** AveRooms (Promedio de Cuartos por Hogar): La distribución de AveRooms presenta una sesgo positivo (derecha) con una gran concentración de valores bajos (por debajo de 20) y una larga cola que se extiende hasta valores mucho más altos. Esto indica que, aunque la mayoría de los hogares tienen pocos cuartos, existen algunos hogares (a lo mejor outliers) con un número mucho mayor, lo que sugiere una gran variabilidad en el tamaño de las viviendas.

4. **AveBedrms (Promedio de Dormitorios):** Al igual que AveRooms, AveBedrms tiene un sesgo positivo (derecha) con una distribución muy concentrada en valores bajos y una cola larga hacia valores más altos. Esto puede reflejar una estructura similar en los hogares, donde la mayoría tiene pocos dormitorios, pero con algunos hogares con más de 30 dormitorios, lo cual es un caso atípico.

5. **Population (Población):** La distribución de Population está claramente sesgada positivamente (derecha), con la mayoría de las observaciones concentradas en poblaciones pequeñas (<5,000) y una cola larga hacia poblaciones más grandes (>30,000). Esto sugiere que la mayoría de las áreas en el dataset son pequeñas, pero algunas tienen una población considerablemente mayor.

6. **AveOccup (Ocupación Promedio):** AveOccup muestra un sesgo muy fuerte hacia la derecha, con una concentración extrema de valores en torno a valores bajos (<200). Hay unos pocos registros con valores mucho mayores, indicando que en ciertos hogares la ocupación es mucho mayor, pero son outliers en la distribución.

7. **Latitude/Longitude:** La distribución de Latitude muestra una forma multimodal con picos en los valores cercanos a 34° y 37-38°. Esto refleja la concentración de los datos en ciertas áreas geográficas de California, como Los Ángeles (34°) y la Bahía de San Francisco (37°). La desviación estándar de latitud baja (2.14) indica poca variabilidad, lo que sugiere que los datos se concentran principalmente en el centro de California. La desviación estándar de longitud (2.00) es baja, lo que refleja que la mayoría de los distritos se encuentran en zonas cercanas.

8. **Price (Target):** Similar a Latitude, Longitude tiene una distribución multimodal, con picos en las longitudes cercanas a -118° (Los Ángeles) y -122° (San Francisco). Esto también refleja una concentración geográfica en ciertas áreas de California, principalmente en las zonas urbanas.


### 3.2 Análisis de Correlación

**Descripción:**

Este análisis examina las relaciones lineales entre todas las variables del dataset mediante la matriz de correlación de Pearson. El objetivo es identificar qué características tienen mayor relación con el precio de las casas y detectar posibles problemas de multicolinealidad entre predictores.

**Resultados Observados:**

![Matriz de Correlación](../_assets/image-03.png)

**Correlaciones con el Precio (ordenadas por magnitud):**

**Conclusiones Principales:**

1. **Predictor dominante:** **MedInc** (r=0.688) es el único predictor con correlación fuerte. Un ingreso medio alto se asocia fuertemente con precios altos, explicando ~47% de la varianza (r²=0.473) por sí solo.

2. **Predictores secundarios:** **AveRooms** (r=0.152) y **HouseAge** (r=0.106) tienen correlaciones débiles pero positivas. Más habitaciones y casas ligeramente más nuevas se asocian con precios mayores.

3. **Multicolinealidad detectada:** **Latitude ↔ Longitude** (r=-0.92) están altamente correlacionadas (capturan geografía conjuntamente). **AveRooms ↔ AveBedrms** (r=0.85) también muestran colinealidad esperada. Esto puede inflar la varianza de los coeficientes estimados.

4. **Variables poco informativas:** **AveOccup** (r=-0.024) y **Population** (r=-0.025) tienen correlaciones prácticamente nulas con el precio. Su utilidad individual es cuestionable, aunque pueden aportar en interacciones.

5. **Efecto geográfico no lineal:** Las correlaciones débiles de **Latitude/Longitude** con precio (-0.144/-0.046) no capturan la realidad de que la ubicación es crucial. Esto confirma que la relación geográfica es **no lineal** (zonas costeras caras vs interiores baratas, independiente de coordenadas absolutas).

---

### 3.3 Entrenamiento y Evaluación del Modelo Múltiple

**Descripción:**

Este punto implementa el entrenamiento completo de un modelo de regresión lineal múltiple con 8 variables predictoras. Se utiliza la división estándar 80/20 para entrenamiento/prueba, con estandarización previa de las características mediante `StandardScaler`. El modelo se evalúa en ambos conjuntos para detectar posibles problemas de overfitting o underfitting.

**Metodología:**
- **División de datos:** 80% entrenamiento, 20% prueba 
- **Preprocesamiento:** Estandarización z-score (media=0, std=1) para cada feature
- **Modelo:** Regresión Lineal OLS (Ordinary Least Squares) sin regularización
- **Random seed:** 42 para reproducibilidad

**Resultados Obtenidos:**

![Evaluación del Modelo](../_assets/image-04.png)

**Métricas Comparativas:**

| Métrica      | Entrenamiento | Prueba  | Diferencia | Interpretación           |
|--------------|---------------|---------|------------|--------------------------|
| **R²**       | 0.6126        | 0.5758  | -0.0368    | Leve degradación         |
| **RMSE**     | 0.7197        | 0.7456  | +0.0259    | Aumento del error (3.6%) |
| **MAE**      | 0.5286        | 0.5332  | +0.0046    | Muy similar              |
| **MSE**      | 0.5179        | 0.5559  | +0.0380    | Aumento moderado         |

**Interpretación Detallada:**

1. **Capacidad Explicativa (R²):**

   **R² Train = 0.6126 (61.26%):**
   - El modelo explica el 61.26% de la variabilidad en los precios del conjunto de entrenamiento
   - Este valor es **moderado**, no excelente. Significa que ~39% de la varianza no es capturada
   - Comparado con el R²=0.94 de regresión simple sintética, aquí enfrentamos **datos reales más complejos**
   
   **R² Test = 0.5758 (57.58%):**
   - En datos no vistos, el modelo explica el 57.58% de la varianza
   - **Degradación de 3.68 puntos porcentuales** respecto al entrenamiento
   - Esta caída es **razonable y esperada** en ML (indica que el modelo no está memorizado)

   El modelo tiene capacidad predictiva real, aunque limitada. La varianza no explicada (42%) sugiere que factores no capturadosinfluyen significativamente en el precio.

2. **Error Absoluto Promedio (MAE):**

   **MAE Train = 0.5286 (±$52,860):**
   - En promedio, las predicciones en entrenamiento se desvían ±$52,860 del precio real
   - Para precios con media $206,000, esto representa un **error del 25.6%**
   - **Contexto:** Una casa de $200,000 podría predecirse entre $147,140 y $252,860
   
   **MAE Test = 0.5332 (±$53,320):**
   - El error promedio en prueba es **casi idéntico** (+$460, solo 0.87% de aumento)
   - **Estabilidad excelente:** El modelo generaliza bien, no está sobreajustado
   - Error consistente entre train/test valida la robustez del modelo

   MAE es más interpretable que MSE. Un error de ~$53k en predicciones de casas de $200k es un factor relevante a considerar, demostrando la busqueda de la mejora de la base que se tiene del modelo.

3. **Error Cuadrático Medio (RMSE):**

   **RMSE Train = 0.7197 (±$71,970):**
   - Raíz del error cuadrático promedio es $71,970
   - **RMSE > MAE** (0.7197 vs 0.5286) por **36%** → indica presencia de **errores grandes ocasionales**
   - El RMSE penaliza más los outliers (errores de $200k impactan 4× más que errores de $100k)
   
   **RMSE Test = 0.7456 (±$74,560):**
   - Aumento de $2,590 respecto al entrenamiento (solo 3.6%)
   - **Diferencia mínima** confirma que el modelo no sufre overfitting significativo
   - RMSE Test < 1.0 se considera **bueno** para este dataset
   

4. **Análisis Visual de las Predicciones:**
- El modelo muestra un **ajuste moderado** en ambos conjuntos, con un **desempeño ligeramente mejor en el entrenamiento**. La dispersión en las predicciones muestra que el modelo podría beneficiarse de mejoras en la capacidad de generalización o de técnicas de regularización para mejorar su ajuste y precisión.


5. **Diagnóstico de Overfitting/Underfitting:**

   **Indicadores evaluados:**
   
   | Indicador                  | Valor Observado | Diagnóstico                    |
   |----------------------------|-----------------|--------------------------------|
   | R² Train - R² Test         | 0.0368 (3.68%)  | **Generalización saludable** |
   | RMSE Test / RMSE Train     | 1.036           | **No overfitting** (<1.1)   |
   | MAE Test ≈ MAE Train       | Δ = 0.0046      | **Estabilidad excelente**   |
   | R² absoluto                | 0.5758 (test)   | **Underfitting moderado**   |
   | Varianza explicada         | 57.58%          | **Capacidad limitada**      |
   
   **Conclusión del diagnóstico:**
   - Las métricas train/test son muy similares para que exista overfitting
   - R²=0.58 sugiere que el modelo es **demasiado simple** para la complejidad del problema
   - El modelo lineal básico **no captura todas las relaciones** en los datos

---

### 3.4 Importancia de Características

**Descripción:**

Este análisis examina los coeficientes estandarizados del modelo para determinar qué variables tienen mayor impacto en las predicciones de precio. Los coeficientes representan el cambio en el precio (en desviaciones estándar) cuando cada feature aumenta en 1 desviación estándar, manteniendo las demás constantes.

**Resultados Obtenidos:**

![Importancia de Características](../_assets/image-05.png)

**Conclusiones Principales:**

1. **Latitude** y **Longitude** son los predictores más importantes (|coef| ≈ 0.87-0.90). Sus coeficientes negativos indican que moverse hacia el norte (mayor latitud) o hacia el este (mayor longitud, menos negativa) **reduce** el precio. Esto captura el efecto de la ubicación costera del sur de California (LA, San Diego) donde los precios son más altos.

2. **MedInc** (Ingreso medio) (coef=+0.854) es el tercer predictor más importante y el **único económico relevante**. Un aumento de 1 std en ingreso medio (~$19k) aumenta el precio en 0.854 std (~$98k), consistente con la correlación r=0.688 observada previamente.

3. **AveBedrms** (coef=+0.339) tiene impacto moderado positivo, mientras que **AveRooms** (coef=-0.294) es negativo. Estos que están inversos podrían demostrar el problema de multicolinealidad desde mi perspectiva, pero se destaca su importancia.

4. **Population** (coef=-0.002) y **AveOccup** (coef=-0.041) son prácticamente **inútiles** para el modelo. Esto podría determinarse como un valor relevante para la eliminación de las features en feature engineering.

5. **Edad de casas poco importante:** **HouseAge** (coef=+0.123) tiene efecto positivo débil, sugiriendo que casas ligeramente más viejas son marginalmente más caras (posiblemente por ubicaciones establecidas), pero el impacto es mínimo.


---

## 4. Gradiente Descendiente
## 4. Gradient Descent desde Cero
Podemos obeservar al graficar nuestra función de coste que tenemos un minimo global para nuestra función convexa en [-0.005 -0.005]

### 4.1 Implementación y Convergencia

**Descripción:**

Esta sección implementa manualmente el algoritmo de descenso del gradiente para resolver regresión lineal mediante optimización iterativa. A diferencia de la ecuación normal (en la que toca invertir matriz y es costoso computaiconalmente), el gradiente descendiente encuentra los parámetros óptimos minimizando iterativamente la función de costo MSE mediante actualizaciones basadas en el gradiente: **θ_new = θ_old - α∇J(θ)**.

**Metodología:**
- **Inicialización:** θ = [0, 0, 0, 0] (intercepto + 3 coeficientes)
- **Learning rate (α):** 0.1
- **Iteraciones:** 500
- **Función de costo:** J(θ) = (1/2m)Σ(ŷ - y)²
- **Dataset sintético:** 100 muestras, 3 features, parámetros verdaderos = [2, -1, 0.5, 3]

**Resultados Obtenidos:**

![Convergencia de Gradient Descent](../_assets/image-06.png)

**Evolución del Costo:**

| Iteración | Costo J(θ) | Cambio vs Anterior | Estado               |
|-----------|------------|--------------------|----------------------|
| 0         | 6.3952     | -                  | Inicial (θ = 0)      |
| 100       | 0.0946     | -6.3006 (-98.5%)   | Convergencia rápida (practicamente alcanzada)  |
| 200       | 0.0946     | ~0.0000            | Convergencia alcanzada|
| 300       | 0.0946     | 0.0000             | Estabilizado         |
| 400       | 0.0946     | 0.0000             | Estabilizado         |

**Parámetros Recuperados:**

| Parámetro    | Valor Verdadero | Valor Encontrado | Error Absoluto | Error Relativo |
|--------------|-----------------|------------------|----------------|----------------|
| **θ₀** (Intercepto) | 2.00   | 2.056            | 0.056          | 2.8%           |
| **θ₁**       | -1.00           | -1.039           | 0.039          | 3.9%           |
| **θ₂**       | 0.50            | 0.475            | 0.025          | 5.0%           |
| **θ₃**       | 3.00            | 2.946            | 0.054          | 1.8%           |

**Análisis de Convergencia:**

1. **Fase Inicial (0-20 iteraciones):**
   - El gráfico muestra una **curva exponencial decreciente pronunciada**, es decir, una caida importante del costo.
   - Los parámetros se actualizan con **gradientes grandes** porque están lejos del óptimo
   - El algoritmo detecta rápidamente la dirección correcta hacia el mínimo

2. **Fase de Convergencia Rápida (20-100 iteraciones):**
   - El costo pasa de 0.50 a 0.0946 (-81%)
   - La curva se aplana gradualmente, indicando que los parámetros se acercan al óptimo
   - Los pasos de actualización disminuyen automáticamente dado que nos estamos acercando al mínimo
   - En el zoom de las primeras 100 iteraciones (gráfico derecho), la curva naranja es casi plana después de la iteración 30

3. **Fase de Estabilización (>100 iteraciones):**
   - **Costo constante:** J(θ) = 0.0946 sin cambios significativos (decimales estables)
   - El algoritmo alcanza el **mínimo global** de la función convexa
   - **Convergencia numérica:** Los gradientes son ~0, actualizaciones <10⁻⁶
   - **Conclusión:** El modelo ha convergido, iteraciones adicionales son innecesarias

4. **Precisión de los Parámetros:**
   - Todos los parámetros tienen error <5.0%, lo que demuestra la efectividad del método y su éxito para encontrar los mismos.

5. **Validación del Learning Rate:**
   ![alt text](../_assets/image-07.png)
   - **α = 0.1 es óptimo** para este problema, dado que como vimos en clase la importancia de considerar el valor óptimo del mismo, podemos concluir que:
     - Si α < 0.01: Convergencia muy lenta (Muchas más iteraciones)
     - Si α > 0.5: Posible divergencia u oscilación alrededor del mínimo
   - La convergencia suave en <100 iteraciones confirma la elección correcta de α para este caso particular.
   - De hecho, este hiper parámetro del paso, podemos encontrar uno que nos de resultados más rápido, como para el ejemplo de "0.1".

   `` No es necesario hacer tantas iteraciones, pues ya sabemos a partir de que punto alcanzamos convergencia. ``

---

## 5. Regularización

### 5.1 Comparación de Técnicas de Regularización

**Descripción:**

Este experimento compara cuatro enfoques para prevenir overfitting en regresión lineal: regresión lineal estándar (OLS), Ridge (L2), LASSO (L1) y Elastic Net (combinación L1+L2). Se utiliza un dataset sintético con **20 features** donde solo **5 son relevantes**, diseñado intencionalmente con **multicolinealidad** (features correlacionadas) para exponer las diferencias entre métodos.

**Metodología:**
- **Dataset:** 100 muestras, 20 features (solo 5 con coeficientes no-cero)
- **Multicolinealidad inducida:** X₁, X₂, X₃ están altamente correlacionadas (r > 0.9)
- **Split:** 70% entrenamiento, 30% prueba
- **Hiperparámetros:** Ridge (α=1.0), LASSO (α=0.1), ElasticNet (α=0.1, l1_ratio=0.5)

**Resultados Obtenidos:**

![Comparación de Regularización](../_assets/image-08.png)

**Tabla de Métricas:**

| Modelo       | Train R² | Test R² | MSE Test | Features ≈ 0 | Sparsity |
|--------------|----------|---------|----------|--------------|----------|
| **Linear**   | 0.989    | 0.979   | 0.355    | 1            | 5%       |
| **Ridge**    | 0.987    | 0.971   | 0.485    | 2            | 10%      |
| **LASSO**    | 0.983    | 0.978   | 0.379    | **17**       | **85%**  |
| **ElasticNet** | 0.982  | 0.973   | 0.455    | 10           | 50%      |

**Interpretación Detallada:**

**Interpretación Detallada:**

1. **Regresión Lineal Estándar (OLS):** Este método logra los mejores resultados en los datos de prueba (R²=0.979, MSE=0.355) usando prácticamente todas las variables disponibles (19 de 20). En el escenario controlado de nuestro experimento, funciona excelentemente. Sin embargo, aquí hay un problema: al intentar usar todo lo que tiene disponible sin discriminar, el modelo se vuelve vulnerable cuando nos enfrentamos a problemas del mundo real con cientos o miles de variables **(sobreajustando)**. La pequeña diferencia entre entrenamiento (R²=0.989) y prueba (R²=0.979) nos dice que afortunadamente no está sobreajustando *en este caso específico*, pero el hecho de que solo elimine 1 variable de 20 nos muestra que no puede distinguir entre información útil y ruido.

2. **Ridge (Regularización L2):** Al comprimir las variables, se sacrifica algo de precisión (MSE aumenta 36% comparado con Linear) a cambio de mayor estabilidad, especialmente cuando hay variables correlacionadas. El problema aquí es que Ridge es *demasiado* conservador para nuestro experimento, donde sabemos que 15 de 20 variables son ruido puro. Al mantener activas todas las variables (solo 10% llegan a cero), termina incorporando varianza innecesaria que empeora las predicciones. Dadas tantas variables irrelevantes, el comprimir no funciona.

3. **LASSO (Regularización L1):** LASSO toma un enfoque radical pero efectivo: mediante la penalización J = MSE + α·Σ|θᵢ|, literalmente elimina variables poniéndolas en cero. En nuestro experimento, borra 17 de 20 variables (85%) y aun así mantiene un desempeño casi perfecto (R²=0.978, MSE=0.379). Perdemos solo 6.8% en precisión pero ganamos un modelo con 85% menos variables, lo que significa menor costo de recolección de datos, inferencias más rápidas, y explicaciones más claras. LASSO identificó correctamente que solo 3 variables importan (cerca de las 5 reales del dataset), creando un buen modelo. Es el método ideal cuando sospechamos que pocas variables contienen la información importante y aquí lo vemos evidenciado.

4. **Elastic Net (L1 + L2):** Elastic Net al combinar las fortalezas de Ridge y LASSO, Con l1_ratio=0.5, logra un balance, pues elimina 50% de variables (10 de 20) con un desempeño intermedio (R²=0.973, MSE=0.455). Su verdadera ventaja aparece cuando hay grupos de variables correlacionadas: mientras LASSO tiende a "elegir caprichosamente" solo una del grupo, Elastic Net las mantiene juntas con coeficientes reducidos, mejorando la estabilidad. En nuestro caso, donde X₁, X₂ y X₃ están altamente correlacionadas (r>0.9), Elastic Net balancea la eliminación de ruido con la conservación de grupos informativos. Es la opción más segura cuando no sabemos de antemano cómo están relacionadas nuestras variables o cuando necesitamos que el modelo sea robusto ante cambios en los datos.


---

## 6. Regresión Logística

### 6.1 Implementación Manual y Evaluación

**Descripción:**

Esta sección implementa desde cero el algoritmo de **Regresión Logística**, el método estándar para problemas de clasificación binaria. A diferencia de la regresión lineal que predice valores continuos, la regresión logística utiliza la **función sigmoide** σ(z) = 1/(1 + e⁻ᶻ) para transformar la combinación lineal de features (z = θᵀx) en una probabilidad entre 0 y 1, interpretable como P(y=1|x). El modelo se entrena mediante Gradient Descent minimizando la **función de costo cross-entropy**: J(θ) = -(1/m)Σ[y·log(σ(θᵀx)) + (1-y)·log(1-σ(θᵀx))], que penaliza fuertemente las predicciones incorrectas con alta confianza.

**Metodología:**
- **Dataset sintético:** 200 muestras (100 por clase) con 2 features, dos clases separadas espacialmente (centroides en [2,2] y [-2,-2])
- **Algoritmo:** Gradient Descent con learning rate α=0.1, 500 iteraciones
- **Inicialización:** θ = [0, 0, 0] (intercepto + 2 coeficientes)
- **Función de activación:** Sigmoide con clipping en [-500, 500] para estabilidad numérica

**Resultados Obtenidos:**

![Regresión Logística](../_assets/image-09.png)

**Parámetros Entrenados:**
- θ₀ (intercepto) = 0.127 → Pequeño desplazamiento de la frontera de decisión desde el origen
- θ₁ (coef. Feature 1) = -1.831 → Feature 1 tiene fuerte peso **negativo** (empuja hacia Clase 0)
- θ₂ (coef. Feature 2) = -1.603 → Feature 2 contribuye similarmente con peso negativo

**Interpretación de los Parámetros:**

Los coeficientes negativos en θ₁ y θ₂ revelan la geometría del problema: dado que la Clase 1 se generó en el cuadrante superior-derecho con centroide en [2,2] y la Clase 0 en el inferior-izquierdo [-2,-2], el modelo aprendió que **valores altos en ambas features aumentan la probabilidad de pertenecer a Clase 1**. Puntos donde x₁ + x₂ < -0.07 tienen probabilidad P(y=1) < 0.5 (región azul, Clase 0), mientras que puntos con x₁ + x₂ > -0.07 tienen P(y=1) > 0.5 (región roja, Clase 1).

La **función sigmoide** transforma la combinación lineal z = θᵀx en probabilidades mediante σ(z) = 1/(1 + e⁻ᶻ): para un punto en [-3, -3], z = 0.127 - 1.831(-3) - 1.603(-3) = 10.43, lo que produce P(y=1) = σ(10.43) ≈ 0.99997 (casi certeza de Clase 1). Esto explica por qué el modelo alcanza **accuracy de 99%**: las clases están bien separadas espacialmente, y solo 1-2 muestras cercanas a la frontera.

**Análisis de Convergencia:**
![Regresión Logística](../_assets/image-10.png)

El gráfico de convergencia revela propiedades clave del problema de optimización:

1. **Problema convexo confirmado:** La función de costo cross-entropy es **estrictamente convexa** (según lo investigado) para regresión logística, garantizando la existencia de un **único mínimo global**. El Gradient Descent encuentra el mínimo sin quedar atrapado en óptimos locales.

2. **Convergencia exponencial:** El costo cae de 0.52 (iteración 0, equivalente a clasificador aleatorio con θ=0) a ~0.015 en las primeras 100 iteraciones, representando una **reducción del 97%**. Este comportamiento exponencial es típico cuando el learning rate α=0.1 está bien calibrado (como bien se habló anteriormente): suficientemente grande para progresar rápido, pero no tan grande que cause inestabilidad (α > 0.5 causaría oscilaciones o divergencia).

3. **Estabilización en mínimo global:** Después de la iteración 100, el costo se estabiliza en ~0.01, indicando que el algoritmo alcanzó el **mínimo global** donde ∇J(θ) ≈ 0. La ausencia de mejoras significativas post-iteración 100 sugiere que 500 iteraciones son más que suficientes.


---

## 7. Métricas de Clasificación

### 7.1 Análisis Completo de Métricas - Dataset Breast Cancer

**Descripción:**

Esta sección evalúa el desempeño de nuestro modelo de **Regresión Logística** aplicado al dataset real de cáncer de mama de Wisconsin, que contiene 569 muestras con 30 características numéricas derivadas de imágenes digitalizadas de biopsias. El objetivo es clasificar tumores como **benignos (Clase 0, n=212)** o **malignos (Clase 1, n=357)**, donde un error de clasificación tiene implicaciones críticas: un **falso negativo** (predecir benigno cuando es maligno) puede retrasar tratamiento vital, mientras que un **falso positivo** (predecir maligno cuando es benigno) genera ansiedad y procedimientos innecesarios. Este desbalance en el costo de errores motiva el análisis exhaustivo de múltiples métricas más allá del accuracy.

**Resultados Obtenidos:**

![Métricas de Clasificación](../_assets/image-11.png)

**Interpretación por Gráfica:**

**1. Matriz de Confusión:**

La matriz revela un desempeño casi perfecto con solo **2 errores en 171 predicciones** (accuracy 98.8%): **1 falso positivo** (tumor benigno clasificado como maligno) y **1 falso negativo** (tumor maligno clasificado como benigno). Los valores diagonales (63 verdaderos negativos, 106 verdaderos positivos) dominan la matriz, indicando alta concordancia entre predicciones y realidad. La **sensibilidad (recall) del 99.07%** para malignos significa que el modelo detecta correctamente 105 de 106 tumores peligrosos, crucial en un contexto médico donde *no detectar un cáncer* es más costoso que una falsa alarma. La **especificidad del 98.44%** confirma que también discrimina bien los casos benignos, evitando tratamientos innecesarios en 63 de 64 pacientes sanos. Se destaca así su capacidad para poder clasificar correctamente inputs para ambas clases. Se destaca la alta calidad de los datos para lograr estas métricas.
 
**2. Curva ROC:**

El **AUC = 0.998** (área bajo la curva casi perfecta) demuestra que el modelo tiene capacidad de discriminar entre clases casi perfecta, siendo capaz de acertar en el 99.8% de los casos la condición del paciente. Simultáneamente se maximiza TPR (True Positive Rate, sensibilidad) y se minimiza FPR (False Positive Rate, 1-especificidad). Este resultado valida que las 30 características del dataset (radio medio, textura, perímetro, área, etc.) contienen **señal predictiva** fuerte y el modelo logró capturarla.

**3. Curva Precision-Recall:**

El **Average Precision = 0.999** confirma desempeño excepcional incluso en el trade-off precision-recall. La curva se mantiene en el extremo superior derecho (precision ≈ 1.0, recall ≈ 1.0), indicando que para casi cualquier umbral de decisión entre 0.1 y 0.9, el modelo mantiene ambas métricas por encima del 95%. Esto es crítico en aplicaciones médicas: podemos ajustar el umbral para priorizar **recall** (detectar todos los casos malignos, tolerando más falsos positivos) o **precision** (evitar falsos positivos, asumiendo riesgo de falsos negativos) sin degradar significativamente la otra métrica.

**4. Distribución de Probabilidades:**

Los histogramas muestran **separación casi perfecta** entre clases: las probabilidades predichas para tumores benignos (azul) se concentran cerca de 0.0 (alta confianza en Clase 0), mientras que para malignos (naranja) se agrupan cerca de 1.0 (alta confianza en Clase 1). La línea vertical negra en 0.5 (umbral por defecto) divide claramente ambas distribuciones, explicando el bajo error. Aquí, la mayoría de predicciones son fiables: P<0.2 o P>0.8.

**5. Métricas vs Umbral**

Este gráfico es fundamental para **ajustar el umbral de decisión** según el contexto del problema. Podemos concluir del mismos que:
- **Precision (rosa)** A medida que el umbral aumenta, la precisión se mantiene relativamente alta (cercana a 1) hasta llegar a un punto crítico. Esto indica que el modelo sigue haciendo buenas predicciones positivas sin generar muchos falsos positivos hasta cierto umbral.

- **Recall (verde)** A medida que el umbral se disminuye, el recall se incrementa rápidamente. Esto indica que el modelo está capturando más de las instancias positivas reales.

**6. Classification Report (Inferior Derecha):**

La tabla cuantifica el desempeño por clase:
- **Clase 0 (Benigno):** Precision 0.984, Recall 0.984, F1 0.984 con 64 muestras → El modelo es igualmente bueno identificando tumores benignos.
- **Clase 1 (Maligno):** Precision 0.991, Recall 0.991, F1 0.991 con 107 muestras → Ligeramente mejor en malignos (posiblemente porque hay más ejemplos de entrenamiento).

El **support** (107 vs 64) muestra leve desbalance hacia malignos en el test set (proporción 1.67:1), pero el modelo no muestra sesgo: ambas clases tienen F1 > 0.98, validando que la regularización L2 y el class_weight='balanced' en LogisticRegression funcionaron correctamente.

**Análisis Crítico:**

Este desempeño excepcional (accuracy 98.8%, AUC 0.998) plantea la pregunta: **¿es realista o hay sobreajuste? (válida pregunta según lo que hemos visto en clase de predicciones tan altas)** Tres evidencias validan que es legítimo:

1. **Dataset de calidad:** Las 30 features del Breast Cancer Wisconsin son mediciones expertas de núcleos celulares (radio, textura, suavidad, compacidad, concavidad, simetría, dimensión fractal) altamente correlacionadas con malignidad según literatura médica.

2. **Separabilidad inherente:** Los tumores malignos tienen características morfológicas distintivas (núcleos más grandes, irregulares, textura heterogénea) que las técnicas de imagen capturan bien. El problema *es* inherentemente separable.

3. **Test set independiente:** El 30% de los datos (171 muestras) se reservó para evaluación, y las métricas en test son comparables a train (no mostradas aquí, pero típicamente Train Accuracy ≈ 99%, Test ≈ 98.8%), indicando buena generalización (asi, evitamos data leakage al separar bien los datos).



---