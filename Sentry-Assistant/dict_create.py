import cv2
from collections import defaultdict
from math import log10

energy_threshold = 0.1
maxlen = 10

def cache_results(face_prediction, confidence, default_dict, energy, learning_rate):
    score = 0
    prev_score = 0
    prev_label = ''

    if(energy > energy_threshold):
        if(confidence > 60):
            values = default_dict.get(face_prediction, None)
            if (values is None):
                default_dict[face_prediction].append(confidence)
            else:
                i = getIndex(values, confidence)
                if(len(values) < maxlen):
                    values.insert(i, confidence)
                else:
                    if(confidence > values[-1]):
                        del values[-1]
                        values.insert(i, confidence)
                    else:
                        energy -= learning_rate * energy + 1
        energy -= 1
    else:
        for label, values in default_dict.items():
            size = len(values)
            score = 10*size + sum(values)/size
            # print("Score: %s" %score +" Label: %s" %label + " Size: %s" %size)
            if(score > prev_score):
                prev_score = score
                prev_label = label
    return prev_label, prev_score, energy

def getIndex(values, confidence):
    index = 0
    for v in values:
        if(confidence <= v):
            index += 1
        else:
            break
    return index

def clearCache(face_prediction):
    return None

# PSEUDO #
#se energy > energy_threshold
#    se default_dict non ha face_prediction come Key
#        default_dict[face_prediction].append(confidence)

#    se default_dict ha già face_prediction come Key
#        estrai la lista di Value:confidence associata alla Key:face_prediction
#        se len(Values) < 10 allora inserisci confidence in posizione i [es. se confidence = 8 --> [14, 13, 10, i, 6, 4]]
#        altrimenti 
#            se confidence > lastElement(Values)
#               allora elimina l'ultimo elemento in Values(il piu basso) e inserisci confidence in posizione i
#            altrimenti decrementa energy
#    decrementa energy

#altrimenti 
#    elimina da default_dict tutte le [Key:Values] per cui len(values) == 1
#    calcola lo score = (Sommatoria(default_dict[Key].Values[i]))/len(Values)
#    memorizza la coppia [face_prediction, score] in scores_list
#    return face_prediction, score from max(score) in scores_list

# ALTERNATIVA - fare l'append a fine lista e poi far il sorting per ogni etichetta
