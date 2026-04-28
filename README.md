# Il Dataset IRIS

Il dataset è stato introdotto in Fisher, R.A. "The use of multiple measurements in taxonomic problems", Annual Eugenics, 7, Part II, 179-188 (1936). Il dataset contiene informazioni su 150 campioni (istanze) di fiori di iris appartenenti a 3 diverse famiglie (classi): iris setosa, iris versicolor e iris virginica. Ci sono 50 campioni per ogni classe. Per ogni campione, il dataset fornisce 4 attributi (caratteristiche): lunghezza del sepalo (cm), larghezza del sepalo (cm), lunghezza del petalo (cm), larghezza del petalo (cm).

## Analisi Esplorativa del Dataset

### Analisi Statistica

#### Statistiche Descrittive per Caratteristica

L'analisi delle distribuzioni monovariate per ogni caratteristica rivela interessanti pattern:

1. **Sepal Length (cm)**: mostra una distribuzione bimodale con una separazione parziale tra le classi. La classe setosa tende ad avere sepali più corti rispetto alle altre due classi.

![Distribuzione Sepal Length](images/hist_sepal_length_cm.png)

2. **Sepal Width (cm)**: la distribuzione è meno discriminante rispetto alle altre caratteristiche, con un'ampia sovrapposizione tra le classi.

![Distribuzione Sepal Width](images/hist_sepal_width_cm.png)

3. **Petal Length (cm)**: questa caratteristica mostra la migliore separazione tra le classi, con la setosa nettamente separata dalle altre due classi.

![Distribuzione Petal Length](images/hist_petal_length_cm.png)

4. **Petal Width (cm)**: simile alla lunghezza del petalo, fornisce una buona separazione tra le classi, particolarmente per la setosa.

![Distribuzione Petal Width](images/hist_petal_width_cm.png)

#### Analisi Bivariata

Le scatter plot bivariate rivelano:
- Una chiara separazione lineare tra la classe setosa e le altre due classi quando si considerano le caratteristiche dei petali
- Le classi versicolor e virginica mostrano una maggiore sovrapposizione, ma sono ancora parzialmente separabili
- La combinazione di lunghezza e larghezza del petalo fornisce la migliore separazione visuale tra tutte e tre le classi

##### Confronto tra Caratteristiche del Sepalo

![Sepal Length vs Sepal Width](images/scatter_sepal_length_cm_sepal_width_cm.png)

Questo grafico mostra come le caratteristiche del sepalo da sole non permettano una separazione netta tra le classi, con particolare sovrapposizione tra versicolor e virginica.

##### Confronto tra Caratteristiche del Petalo

![Petal Length vs Petal Width](images/scatter_petal_length_cm_petal_width_cm.png)

Le dimensioni del petalo mostrano una separazione eccellente tra tutte e tre le classi, con la setosa completamente separata e versicolor/virginica distinguibili.

##### Confronti Misti Sepalo-Petalo

![Sepal Length vs Petal Length](images/scatter_sepal_length_cm_petal_length_cm.png)

![Sepal Length vs Petal Width](images/scatter_sepal_length_cm_petal_width_cm.png)

![Sepal Width vs Petal Length](images/scatter_sepal_width_cm_petal_length_cm.png)

![Sepal Width vs Petal Width](images/scatter_sepal_width_cm_petal_width_cm.png)

Questi grafici dimostrano che combinando caratteristiche di sepalo e petalo si ottiene comunque una buona separazione, confermando che le caratteristiche del petalo sono le più discriminanti.

### Analisi Statistica delle Medie

#### Medie Globali e per Classe

**Media globale (su tutti i campioni)**:
```
[5.843, 3.057, 3.758, 1.199] cm
```

**Media per classe Setosa** (classe 0):
```
[5.006, 3.428, 1.462, 0.246] cm
```

**Media per classe Versicolor** (classe 1):
```
[5.936, 2.770, 4.260, 1.326] cm
```

Osservazioni:
- La classe setosa ha sepali più corti ma più larghi e petali significativamente più piccoli
- La classe versicolor mostra valori intermedi per la maggior parte delle caratteristiche
- Le caratteristiche dei petali (lunghezza e larghezza) mostrano le differenze più marcate tra le classi

### Matrici di Covarianza

#### Covarianza Globale
```
[[ 0.681, -0.042,  1.266,  0.513]
 [-0.042,  0.189, -0.327, -0.121]
 [ 1.266, -0.327,  3.096,  1.287]
 [ 0.513, -0.121,  1.287,  0.577]]
```

#### Covarianza per Classe Setosa
```
[[0.122, 0.097, 0.016, 0.010]
 [0.097, 0.141, 0.011, 0.009]
 [0.016, 0.011, 0.030, 0.006]
 [0.010, 0.009, 0.006, 0.011]]
```

#### Covarianza per Classe Versicolor
```
[[0.261, 0.083, 0.179, 0.055]
 [0.083, 0.097, 0.081, 0.040]
 [0.179, 0.081, 0.216, 0.072]
 [0.055, 0.040, 0.072, 0.038]]
```

### Osservazioni sulla Covarianza

1. **Varianza**: Le caratteristiche dei petali (elementi diagonali [2,2] e [3,3]) mostrano una varianza molto maggiore rispetto alle caratteristiche dei sepali, indicando una maggiore variabilità nelle dimensioni dei petali tra i campioni.

2. **Correlazione**: Esiste una forte correlazione positiva tra lunghezza e larghezza del petalo (elemento [2,3]), suggerendo che petali più lunghi tendono ad essere anche più larghi.

3. **Omogeneità delle Classi**: La classe setosa mostra valori di varianza molto più bassi rispetto alla classe versicolor, indicando una maggiore omogeneità all'interno della classe setosa.

4. **Separabilità**: La differenza significativa nelle matrici di covarianza tra le classi suggerisce che un classificatore che consideri la struttura di covarianza (come un classificatore Gaussiano quadratico) potrebbe funzionare meglio di un classificatore lineare semplice.

## Analisi con PCA e LDA

### Principal Component Analysis (PCA)

La PCA è stata applicata al dataset completo (3 classi) per ridurre la dimensionalità a 2 componenti principali. Le componenti principali catturano le direzioni di massima varianza nei dati.

![PCA Projection](images/pca_projection.png)

**Osservazioni dalla proiezione PCA (2 componenti)**:
- La prima componente principale cattura la maggior parte della varianza totale
- Nonostante la riduzione dimensionale, le tre classi rimangono visivamente separabili
- La classe setosa si separa nettamente dalle altre due lungo la prima componente
- Le classi versicolor e virginica mostrano una maggiore sovrapposizione rispetto alla visualizzazione con le caratteristiche originali

### Linear Discriminant Analysis (LDA)

L'LDA è stata applicata per trovare le direzioni che massimizzano la separazione tra le classi, minimizzando la varianza intra-classe e massimizzando quella inter-classe.

**Matrici di Covarianza Calcolate**:

**Between-class covariance matrix (Sb)**:
```
[[ 0.421, -0.133,  1.102,  0.475]
 [-0.133,  0.076, -0.382, -0.153]
 [ 1.102, -0.382,  2.914,  1.245]
 [ 0.475, -0.153,  1.245,  0.536]]
```

**Within-class covariance matrix (Sw)**:
```
[[0.260, 0.091, 0.164, 0.038]
 [0.091, 0.113, 0.054, 0.032]
 [0.164, 0.054, 0.181, 0.042]
 [0.038, 0.032, 0.042, 0.041]]
```

**Osservazioni dalla proiezione LDA (2 componenti)**:
- L'LDA produce una separazione ottimale tra le classi rispetto alla PCA
- Le tre classi formano cluster distinti e ben separati
- La direzione discriminante principale separa perfettamente la setosa dalle altre due classi
- La seconda direzione discriminante aiuta a separare versicolor e virginica

![LDA Projection](images/lda_projection.png)

Si può osservare come l'LDA, essendo supervisionata, produca una separazione molto più netta rispetto alla PCA. I cluster delle tre classi sono chiaramente distinguibili nel piano LDA.

## Classificazione Binaria: Versicolor vs Virginica

### Preparazione del Dataset

Per valutare le performance di classificazione, è stato creato un dataset binario contenente solo le classi Versicolor (label 1) e Virginica (label 2), eliminando la classe Setosa:
- **Campioni totali**: 100 (50 per classe)
- **Split**: 2/3 training (66 campioni), 1/3 validation (34 campioni)
- **Seed casuale**: 0 (per riproducibilità)

### Classificatore LDA Puro

Il classificatore LDA è stato addestrato sui dati di training per trovare la direzione discriminante ottimale tra le due classi.

**Risultati**:
- **Threshold**: 4.079
- **Media Versicolor proiettata**: 2.131
- **Media Virginica proiettata**: 6.026
- **Errori**: 2 su 34 campioni
- **Error rate**: 5.9%

**Analisi**:

#### Visualizzazione delle Distribuzioni

Gli istogrammi dei campioni proiettati mostrano la separazione tra le classi nel training set e validation set:

![LDA Binary Histograms](images/lda_binary_histograms.png)

Si può osservare come:
- Le distribuzioni delle due classi (Versicolor in arancione e Virginica in verde) siano ben separate
- Il training set e il validation set mostrano distribuzioni simili, indicando assenza di overfitting
- Esiste una piccola sovrapposizione tra le code delle due distribuzioni, che causa i 2 errori di classificazione
- Il threshold (circa 4.08) si trova in una posizione ottimale tra le due distribuzioni
- L'LDA riesce a separare efficacemente le due classi con un basso tasso di errore
- Le medie proiettate sono ben separate (differenza di ~3.9), indicando una buona discriminazione
- Il threshold calcolato come media delle medie di classe è appropriato per la classificazione
- Gli errori (5.9%) sono principalmente dovuti alla sovrapposizione naturale tra le due classi nella regione di confine

### Classificatore LDA con Pre-processing PCA

Per valutare l'effetto della riduzione dimensionale, è stato applicato PCA come pre-processing prima dell'LDA, testando diverse dimensionalità.

**Risultati per diverse dimensioni PCA**:

| Dimensioni PCA (m) | Threshold | Mean Versicolor | Mean Virginica | Errori | Error Rate |
|-------------------|-----------|-----------------|----------------|--------|------------|
| m=1 | 10.885 | 9.874 | 11.897 | 4 | 11.8% |
| m=2 | 5.354 | 3.591 | 7.118 | 2 | 5.9% |
| m=3 | 4.535 | 2.745 | 6.324 | 2 | 5.9% |
| m=4 | 4.079 | 2.131 | 6.026 | 2 | 5.9% |

**Analisi dei Risultati**:

1. **m=1 (Prima componente PCA)**:
   - Performance peggiore: 11.8% error rate (4 errori)
   - La prima componente PCA cattura la massima varianza ma non necessariamente la migliore direzione discriminante
   - Conferma che massimizzare la varianza (PCA) non equivale a massimizzare la separabilità delle classi (LDA)

2. **m=2 (Due componenti PCA)**:
   - Performance ottimale: 5.9% error rate (2 errori)
   - Risultato identico all'LDA senza PCA
   - Due dimensioni sono sufficienti per catturare tutta l'informazione discriminante

3. **m=3 (Tre componenti PCA)**:
   - Performance ottimale: 5.9% error rate (2 errori)
   - Nessun miglioramento rispetto a m=2
   - La terza dimensione non aggiunge informazione discriminante significativa

4. **m=4 (Tutte le features)**:
   - Performance identica all'LDA senza PCA (come atteso)
   - Conferma che mantenere tutte le dimensioni preserva completamente l'informazione

**Osservazioni Chiave**:

- **PCA vs LDA**: La prima componente PCA non è ottimale per la classificazione, confermando che PCA (non supervisionato) e LDA (supervisionato) ottimizzano obiettivi diversi
- **Riduzione Dimensionale Efficace**: Due dimensioni PCA sono sufficienti per mantenere le stesse performance dell'LDA su 4 dimensioni
- **Trade-off Complessità/Performance**: L'uso di PCA+LDA con m=2 o m=3 riduce la dimensionalità del 50-25% senza perdita di accuratezza
- **Orientamento Critico**: Per entrambi i classificatori è stato necessario verificare e correggere l'orientamento della direzione discriminante per garantire che la classe Virginica avesse media proiettata maggiore

### Validazione del Modello

La strategia di split train/validation ha permesso di:
- Valutare le performance su dati non visti durante il training
- Evitare overfitting (le performance su validation sono simili a quelle attese su training)
- Confrontare oggettivamente diverse configurazioni di classificatori
- Simulare uno scenario di applicazione reale

## Conclusioni

### Analisi Esplorativa
L'analisi esplorativa del dataset IRIS rivela che:
- Le caratteristiche dei petali sono altamente discriminanti per la classificazione
- La classe setosa è ben separata dalle altre due classi
- Le classi versicolor e virginica sono più difficili da separare, ma mostrano pattern distinguibili
- Le relazioni multivariate tra le caratteristiche suggeriscono che approcci di machine learning che sfruttano queste correlazioni potrebbero ottenere buone performance di classificazione

### Tecniche di Dimensionality Reduction e Classificazione
L'applicazione di PCA e LDA ha dimostrato che:
- **LDA è superiore a PCA per la classificazione**: L'LDA, essendo supervisionato, trova direzioni ottimali per la separazione delle classi
- **PCA può essere efficace come pre-processing**: Con m≥2, PCA+LDA raggiunge le stesse performance di LDA puro riducendo la dimensionalità
- **La prima componente PCA non è sufficiente**: Massimizzare la varianza non garantisce massima separabilità delle classi
- **Efficienza computazionale**: La riduzione a 2-3 dimensioni può accelerare significativamente i classificatori più complessi mantenendo l'accuratezza

### Performance di Classificazione
I risultati del problema binario (Versicolor vs Virginica) mostrano:
- **Eccellente separabilità**: 5.9% error rate è un ottimo risultato considerando la sovrapposizione naturale delle classi
- **Robustezza**: Le performance sono consistenti tra diverse configurazioni (LDA puro, PCA+LDA con m≥2)
- **Validazione efficace**: Lo split train/validation ha permesso una valutazione realistica delle performance

Questi risultati confermano che il dataset IRIS è ideale per l'apprendimento e il testing di algoritmi di classificazione, fornendo un benchmark chiaro e ben strutturato per tecniche di machine learning.

## Appendice: Osservazioni sui Grafici con Fit Gaussiano per Classe

In questa sezione vengono riportate le osservazioni ottenute dai grafici che sovrappongono, per ciascuna feature, l'istogramma empirico e il fit Gaussiano monodimensionale per classe.

### 1) Sepal Length (cm)

![Fit Gaussiano Sepal Length](images/gaussian_fit_sepal_length.png)

Osservazioni:

- La classe **setosa** è centrata su valori più bassi rispetto alle altre due classi.
- **versicolor** e **virginica** risultano parzialmente sovrapposte nella zona centrale.
- Il fit Gaussiano segue l'andamento generale, ma la separazione non è netta tra le due classi non-setosa.


### 2) Sepal Width (cm)

![Fit Gaussiano Sepal Width](images/gaussian_fit_sepal_width.png)

Osservazioni:

- È la feature meno discriminante: le tre classi mostrano ampia sovrapposizione.
- Le curve Gaussiane hanno picchi in zone vicine, confermando la difficoltà di separazione usando solo questa variabile.


### 3) Petal Length (cm)

![Fit Gaussiano Petal Length](images/gaussian_fit_petal_length.png)

Osservazioni:

- La classe **setosa** è chiaramente separata dalle altre due.
- **versicolor** e **virginica** mantengono una sovrapposizione limitata, soprattutto nelle code.
- Questa feature è altamente informativa per la classificazione.


### 4) Petal Width (cm)

![Fit Gaussiano Petal Width](images/gaussian_fit_petal_width.png)

Osservazioni:

- Anche in questo caso **setosa** è ben separata.
- La sovrapposizione tra **versicolor** e **virginica** è presente ma contenuta.
- La feature conferma un'elevata capacità discriminante, in linea con i grafici bivariati.


### Nota metodologica

Il codice usato per questi grafici è corretto dal punto di vista matematico (stima di media/covarianza e valutazione della densità Gaussiana). Tuttavia, usando range e bin definiti separatamente per ciascuna classe, il confronto visivo tra classi può risultare meno uniforme. Per una visualizzazione più confrontabile, si possono usare bin e asse x comuni per tutte le classi della stessa feature.

## Osservazioni sui Classificatori Gaussiani 

### 1) Classificatore Gaussiano Multivariato (MVG)

Per ogni classe $c$ si stima:

$$
\mu_c = \frac{1}{N_c}\sum_{i=1}^{N_c} x_i, \qquad
\Sigma_c = \frac{1}{N_c}\sum_{i=1}^{N_c}(x_i-\mu_c)(x_i-\mu_c)^T
$$

La densità condizionata è:

$$
f(x\mid c)=\mathcal{N}(x\mid \mu_c,\Sigma_c)
$$

e la decisione multiclasse usa il massimo a posteriori:

$$
\hat c = \arg\max_c \; P(c\mid x) \propto f(x\mid c)P(c)
$$

**Risultato :**

- Error rate: **4.0%**
- Accuracy: **96.0%**

**Osservazione:** il modello MVG è già molto competitivo su IRIS; la separazione tra classi è buona, con errori residui concentrati nelle regioni di sovrapposizione (soprattutto versicolor/virginica).

### 2) Naive Bayes Gaussiano

Rispetto al MVG, impone indipendenza condizionata tra feature dato $c$, usando una covarianza diagonale:

$$
\Sigma_c^{NB}=\operatorname{diag}(\Sigma_c)
$$

Le posteriori si calcolano nello stesso schema MAP.

**Risultato :**

- Error rate: **4.0%**
- Accuracy: **96.0%**

**Osservazione:** su questo split, la semplificazione diagonale non degrada le performance rispetto a MVG; ciò suggerisce che una parte rilevante dell’informazione discriminante è già contenuta nelle statistiche univariate/diagonali.

### 3) Classificatore Gaussiano a Covarianza Legata (Tied Covariance)

Assume una sola matrice di covarianza condivisa tra classi:

$$
\Sigma_c = \Sigma, \qquad
\Sigma = \frac{1}{N}\sum_c\sum_{i=1}^{N_c}(x_{c,i}-\mu_c)(x_{c,i}-\mu_c)^T
$$

Le medie restano specifiche di classe, la covarianza è comune.

**Risultato :**

- Error rate: **2.0%**
- Accuracy: **98.0%**

**Osservazione:** è il miglior risultato tra i tre modelli multiclasse gaussiani testati. La stima condivisa della covarianza riduce la varianza di stima dei parametri e, in questo caso, migliora la generalizzazione.

## Osservazioni sui classificatori binari (Versicolor vs Virginica)


Nel caso binario, il punteggio teorico naturale è il log-likelihood ratio (LLR):

$$
s(x)=\log\frac{f(x\mid C=2)}{f(x\mid C=1)}
=\log f(x\mid C=2)-\log f(x\mid C=1)
$$

Con priori uguali, la soglia teorica è $t=0$ e la decisione è: classe 2 se $s(x)\ge t$, altrimenti classe 1.

### Confronto finale tra i tre classificatori binari

Dalle ultime esecuzioni nel notebook:

- **MVG binario**: error rate **8.82%**, accuracy **91.18%**
- **Naive Bayes binario**: error rate **11.76%**, accuracy **88.24%**
- **Tied Covariance binario**: error rate **5.88%**, accuracy **94.12%**

Osservazioni conclusive:

1. **Tied Covariance** è il modello migliore tra i tre nel problema versicolor vs virginica.
2. **MVG** ottiene prestazioni intermedie: buona separazione, ma con più errori nella zona di sovrapposizione tra le due classi.
3. **Naive Bayes** è il meno accurato in questo split: l'ipotesi di indipendenza condizionata (covarianza diagonale) risulta troppo restrittiva.
4. L'error rate di **Tied Covariance** (**5.88%**) è sostanzialmente uguale a quello osservato per **LDA** (circa **5.9%**), in accordo con la teoria della frontiera lineare con covarianza condivisa.

In sintesi, nel task binario la variante a covarianza legata è la scelta più robusta tra i tre classificatori gaussiani considerati.

## Osservazioni su Logistic Regression 

Riportiamo le osservazioni emerse nel notebook part5.ipynb dedicato alla regressione logistica binaria sul dataset Iris (classi: **versicolor = 1** e **virginica = 0**, con **setosa esclusa**).

Sono stati valutati due modelli:

1. **Logistic Regression standard** con regolarizzazione $\lambda \in \{10^{-3}, 10^{-1}, 1\}$
2. **Logistic Regression pesata (prior-weighted)** con prior target $\pi_T=0.8$

Le metriche considerate sono: error rate, min DCF e actual DCF.

### Risultati: Logistic Regression standard

| $\lambda$ | $J^*$ | Error Rate | min DCF ($\pi=0.5$) | actual DCF ($\pi=0.5$) |
| --- | ---: | ---: | ---: | ---: |
| $10^{-3}$ | $1.100009\times10^{-1}$ | 8.8% | 0.0625 | 0.1181 |
| $10^{-1}$ | $4.539407\times10^{-1}$ | 11.8% | **0.0556** | **0.1111** |
| $1$ | $6.316436\times10^{-1}$ | 14.7% | 0.1111 | 0.1667 |

**Osservazioni principali (standard):**

- Il compromesso migliore in termini di rischio decisionale è con $\lambda=10^{-1}$ (min DCF e actual DCF più bassi).
- $\lambda=10^{-3}$ ottiene l’error rate più basso, ma con DCF leggermente peggiore rispetto a $\lambda=10^{-1}$.
- Aumentando troppo la regolarizzazione ($\lambda=1$), il modello sottoperforma su tutte le metriche.

### Risultati: Logistic Regression pesata ($\pi_T=0.8$)

| $\lambda$ | $J^*$ | Error Rate | min DCF ($\pi_T=0.8$) | actual DCF ($\pi_T=0.8$) |
| --- | ---: | ---: | ---: | ---: |
| $10^{-3}$ | $9.401035\times10^{-2}$ | 11.8% | 0.1667 | **0.2222** |
| $10^{-1}$ | $3.606261\times10^{-1}$ | 38.2% | **0.0556** | 0.7222 |
| $1$ | $4.724715\times10^{-1}$ | 52.9% | 0.1111 | 1.0000 |

**Osservazioni principali (weighted):**

- Il training pesato con $\pi_T=0.8$ rende il classificatore molto sensibile alla calibrazione della soglia: si osserva forte gap tra min DCF e actual DCF, soprattutto per $\lambda=10^{-1}$ e $\lambda=1$.
- Sebbene con $\lambda=10^{-1}$ il min DCF sia basso, l’actual DCF è molto alto: segnale di score poco calibrati rispetto al punto operativo scelto.
- Tra le configurazioni pesate provate, $\lambda=10^{-3}$ è quella più stabile in termini di actual DCF.

Nel setup considerato, la **logistic regression standard** risulta complessivamente più robusta della versione pesata in assenza di una calibrazione aggiuntiva dei punteggi. In particolare, $\lambda=10^{-1}$ (standard) offre il miglior trade-off decisionale, mentre la versione weighted evidenzia che ottimizzare solo il training prior-weighted non garantisce automaticamente buone performance operative (actual DCF) senza un tuning/correct calibration della decisione finale.

## Osservazioni su SVM (notebook part6)

Nel notebook SVM è stato considerato il problema binario Versicolor vs Virginica con split 2/3 training e 1/3 validation, valutando:

- **SVM lineare** (duale con bias regolarizzato tramite estensione con $K$)
- **SVM kernel polinomiale** con $k(x_i,x_j) = (x_i^T x_j + c)^d + \xi$
- **SVM kernel RBF** con $k(x_i,x_j) = \exp(-\gamma\|x_i-x_j\|^2) + \xi$

dove $\xi = K^2$.

### Risultati SVM lineare

| K | C | Error Rate | min DCF ($\pi=0.5$) | actual DCF ($\pi=0.5$) |
| ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.1 | **2.9%** | 0.0556 | 0.0625 |
| 1.0 | 1.0 | 5.9% | 0.0556 | 0.1181 |
| 1.0 | 10.0 | 5.9% | 0.0556 | 0.1181 |
| 10.0 | 0.1 | 11.8% | **0.0000** | 0.2222 |
| 10.0 | 1.0 | 5.9% | 0.0556 | 0.1181 |
| 10.0 | 10.0 | 5.9% | 0.0625 | 0.1181 |

Osservazioni:

- La configurazione più stabile in termini complessivi è **K=1.0, C=0.1** (error rate più basso e actual DCF contenuto).
- Il caso **K=10.0, C=0.1** mostra un esempio di disallineamento tra min DCF e actual DCF: min DCF ottimo ma decisione operativa peggiorata (actual DCF alto).

### Risultati SVM kernel polinomiale ($d=2$)

| K | C | c | Error Rate | min DCF | actual DCF |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 1.0 | 0.0 | 8.8% | 0.0625 | 0.1736 |
| 1.0 | 1.0 | 0.0 | 8.8% | 0.0625 | 0.1736 |
| 0.0 | 1.0 | 1.0 | **2.9%** | **0.0556** | **0.0556** |
| 1.0 | 1.0 | 1.0 | **2.9%** | **0.0556** | **0.0556** |

Osservazioni:

- L'offset del kernel è determinante: con **c=1** le performance migliorano nettamente rispetto a **c=0**.
- In queste prove, il kernel polinomiale con **c=1** e $d=2$ risulta tra le configurazioni migliori dell'intero notebook.

### Risultati SVM kernel RBF

| K | C | $\gamma$ | Error Rate | min DCF | actual DCF |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 1.0 | 1.0 | 8.8% | 0.0625 | 0.1736 |
| 1.0 | 1.0 | 1.0 | 8.8% | 0.0625 | 0.1736 |
| 0.0 | 1.0 | 10.0 | 11.8% | 0.1736 | 0.2292 |
| 1.0 | 1.0 | 10.0 | 8.8% | 0.1736 | 0.1736 |

Osservazioni:

- Nel setup testato, RBF non supera il polinomiale e non migliora la lineare migliore.
- Aumentare troppo $\gamma$ (es. 10) peggiora il comportamento, coerentemente con un kernel troppo "stretto" e minore capacità di generalizzazione.

### Conclusioni SVM

1. Le migliori prestazioni osservate sono con:
   - **SVM lineare**: $K=1.0, C=0.1$
   - **SVM polinomiale**: $d=2, c=1$ (con $K\in\{0,1\}$)
2. Nel notebook analizzato, il **kernel polinomiale con c=1** è quello che fornisce il miglior compromesso complessivo tra error rate, min DCF e actual DCF.
3. Il **kernel RBF** richiede un tuning più fine di $\gamma$ e $C$ per risultare competitivo su questo split.

## Osservazioni su GMM (notebook part7)

Nel notebook part7 è stato implementato un classificatore **GMM multiclasse** (3 classi IRIS) con:

- training separato di un GMM per ciascuna classe;
- algoritmo **LBG** per aumentare progressivamente il numero di componenti;
- raffinamento con **EM vincolato** (soglia sugli autovalori tramite parametro $\psi$);
- decisione finale tramite massimo della log-densità tra classi.

Sono stati testati i seguenti numeri di componenti:

$$
\{1,2,4,8,16\}
$$

### Risultati osservati

|Componenti GMM|Error Rate|
|---:|---:|
|1|4.00%|
|2|4.00%|
|4|4.00%|
|8|4.00%|
|16|4.00%|

### Commento

1. Le performance sono **stabili** al variare del numero di componenti: in questo split l'errore resta al 4.00%.
2. Aumentare la complessità del modello (più componenti) **non porta miglioramenti** misurabili.
3. Questo comportamento è coerente con IRIS: il dataset è relativamente semplice e già ben modellabile con Gaussiane poco complesse.
4. Il vincolo di regolarizzazione sulla covarianza ($\psi$) aiuta a mantenere stabile l'ottimizzazione EM ed evita degenerazioni numeriche.

In sintesi, nel setup testato, il GMM offre buone prestazioni ma non mostra vantaggi rispetto a configurazioni più semplici quando si aumenta il numero di componenti.
