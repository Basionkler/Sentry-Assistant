# 3.0 - Sentry-Wizard
Sistema di Face-recognition e sentiment analysis con stima del rischio per simulazione di sistemi di difesa e anti-terrorismo.

## 3.1 Introduzione
Tra lo studio ed i test effettuati, la versione **Sentry-Wizard** è la più stabile tra le tre. In questa distribuzione abbiamo eliminato del tutto la necessità di avere due reti neurali distinte per analizzare l'espressione facciale e l'identità del soggetto. Il file `sentry_wizard.py` contiene per intero tutta la meccanica di rilevamento video, sfruttando la *webcam* integrata del pc, simulando un flusso dati ottenibile online (eg. *webcam per il traffico*, *webcam nelle stazioni*, *ecc...*). 

## 3.2 Set-Up e Training
Assumendo che abbiate i **Requirements** elencati, è necessario **creare** (nel caso non siano già presenti) le seguenti cartelle. Per farlo basterà eseguire i seguenti comandi:

```
cd Sentry-Wizard
mkdir face_dataset emotion_dataset prepared_faces_dataset trainedmodel
```
Lanciare ora il seguente comando per creare la struttura di addestramento per le **emozioni**:
```
cd emotion_dataset
mkdir raw prepared raw/afraid raw/angry raw/disgusted raw/happy raw/neutral raw/sad raw/surprised
cd ..
```
la cartella `face_dataset` conterrà tutto l'insieme di dati su cui vorrete effettuare il *training* del sistema. Ogni immagine dovrà essere contenuta in una cartella la cui etichetta è il **nome** della persona che vorrete aggiungere. La struttura da adottare è la seguente:
```
*face_dataset/
|----name_1/
     |----1.jpg
     |----2.jpg
     ....
|----name_2/
     |----1.jpg
     |----2.jpg
     ....
```

una volta che il vostro training set sarà pronto e **conforme** alla struttura sopracitata, potrete lanciare il file di elaborazione delle immagini, che si preoccuperà di *ritagliare* automaticamente il volto di ogni singola immagine e portarlo in *scala di grigi* (NB. in questa fase ancora il sistema non sarà addestrato). Basterà lanciare:
```
python id_data_prep.py
```
a questo punto, nella cartella `prepared_faces_dataset` dovreste avere la stessa struttura presente in `face_dataset`, ma elaborata. Queste saranno le immagini su cui il vostro sistema verrà addestrato. A questo punto potrete lanciare il file di **addestramento**:
```
python lbphrecognizer_train.py
```
Il sistema verrà addestrato e verrà generato un file **.yml** in `trainedmodel`. Addestriamo ora il sistema per riconoscere le emozioni. Nella cartella `emotion_dataset` ci saranno due cartelle:

*	prepared
*	raw

In `prepared` verranno generate tutte le foto elaborate dal sistema su cui faremo l'addestramento. In `raw` ci sono le sottocartelle, che abbiamo creato con i comandi precedenti, in cui inserirete il vostro **training-set personalizzato**.

**NB. Per comodità il sistema carica di default un file *.xml* pre-addestrato, presente nella cartella** `models`.

Una volta preparate le emozioni che volete addestrare nelle rispettive cartelle, basterà lanciare i seguenti comandi:
```
python emotion_data_prep.py
python train_emotion_classifier.py
```
in `models` verrà generato(sovrascritto) il file *emotion_classifier_model.xml*, che verrà caricato automaticamente al lancio del programma. Nel caso non siate soddisfatti dell'attuale livello di addestramento, potete rinominare (o usare) il file *emotion_classifier_model_decent.xml*.

## 3.3 dict_create & face_detection
In questa sezione verrà discusso brevemente il funzionamento di questi due file. in `face_detection.py` sono presenti i criteri con i quali vengono identificati i volti all'interno delle diverse immagini (o flusso video). Se necessario è possibile cambiare i parametri presenti per avere una detection più o meno sensibile.

in `dict_create.py` abbiamo implementato un sistema di caching per migliorare i risultati ottenuti con le precedenti versioni. Inizialmente il sistema faceva un confronto, per ogni frame, tra l'immagine ottenuta e tutte le immagini del database, restituendo quella che aveva la **minor distanza tra istogrammi**. Questo era un risultato molto incerto e troppo variabile che ci portava ad ottenere un cambio di identità quasi per ogni frame. Abbiamo dunque inserito un sistema di **caching** in cui creare una mappatura tra **labels** e **lista di confidence**. Basandosi sul principio che usa l'algoritmo **simulated-annealing**, si parte con una certa energia di partenza che decrementa ad ogni passaggio. Per ogni frame analizzato, se la confidence è superiore ad una certa soglia, viene inserita l'immagine in una lista associata alla relativa etichetta. Eseguendo questo processo fino ad esaurimento dell'energia, viene generata una mappa **chiave/lista_di_valori**. Questi dati vengono elaborati per calcolare uno **score** e restituire l'etichetta con lo score più alto. Se lo score è più basso di una soglia scelta (modificabile), allora il soggetto non verrà riconosciuto e verrà visualizzato **Unknown** come identità.

## 3.4 Avvio del programma
Una volta creati i riferimenti citati in **3.2**, sarà possibile lanciare il programma e verificarne le prestazioni ed il funzionamento:

```
python sentry_wizard.py
```
