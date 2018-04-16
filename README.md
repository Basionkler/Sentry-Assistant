# 3.0 - Sentry-Wizard
Sistema di Face-recognition e sentiment analysis con stima del rischio per simulazione di sistemi di difesa e anti-terrorismo.

## 3.1 Introduzione
Tra lo studio ed i test effettuati, la versione **Sentry-Wizard** è la più stabile tra le tre. In questa distribuzione abbiamo eliminato del tutto la necessità di avere due reti neurali distinte per analizzare l'espressione facciale e l'identità del soggetto. Il file `sentry_wizard.py` contiene per intero tutta la meccanica di rilevamento video, sfruttando la *webcam* integrata del pc, simulando un flusso dati ottenibile online (eg. *webcam per il traffico*, *webcam nelle stazioni*, *ecc...*). 

## 3.2 Funzionamento
Assumendo che abbiate i **Requirements** elencato, è fortemente consigliato **creare** (nel caso non siano già presenti). Per farlo basterà eseguire i seguenti comandi:

```
cd Sentry-Wizard
```

```
mkdir dataset prepared_faces_dataset trainedmodel
```


