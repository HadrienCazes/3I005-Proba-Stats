#CAZES Hadrien , SY Virak Bot Baramy
#3I005 Projet 3 Bioinfo

import matplotlib.pyplot as plt
import io
import math
import random
import numpy as np
import scipy.stats as sts

#Définition des nucléotides
nucleotide = {'A':0,'C':1,'G':2,'T':3}
nucleotide_indetermine = {'A':0,'C':1,'G':2,'T':3,'N':-1}

def decode_sequence(sequence):
    inv_nucleotide = {v:k for k, v in nucleotide_indetermine.items()}
    to_str = ""
    for i in sequence:
        if(i in inv_nucleotide):
            to_str += inv_nucleotide[i]
        else:
            to_str += 'N'
    return to_str
    

def encode_sequence(string):
    to_list = []
    for base in string:
        if(base in nucleotide_indetermine):
            to_list.append(nucleotide_indetermine[base])
    return to_list

def read_fasta(fasta_filepath):
    fasta_file = io.open(fasta_filepath, 'r')
    current_sequence = ""
    sequences_dict = {}
    for line in fasta_file.readlines():
        if(line[0] == '>'):
            current_sequence = line
            sequences_dict[line] = []
        else:
            for nucl in line:
                if(nucl in nucleotide_indetermine):
                    sequences_dict[current_sequence].append(nucleotide_indetermine[nucl])

    return sequences_dict
    
def nucleotide_count(sequence):
    count = [0 for k in nucleotide]
    for nucl in sequence:
        if(nucl >= 0):
            count[nucl] += 1
    return count

def nucleotide_frequency(sequence):
    count = [0 for k in nucleotide]
    n_nucl = 0.
    for nucl in sequence:
        if(nucl >= 0):
            count[nucl] += 1
            n_nucl += 1.
    return count/(np.sum(count))

def logproba(sequence, m):
    count = nucleotide_count(sequence)
    res = 0.0
    for i in range(len(count)):
        res += count[i] * math.log(m[i])
    return res

def logprobafast(compte_lettres, m):
    res = 0.0
    for i in range(len(compte_lettres)):
        res += compte_lettres[i] * math.log(m[i])
    return res

def code(m, k):
    """
    conversion base 4 à base 10
    """
    res = 0
    j=k
    for i in m:
        res = res + i*(4**(j-1))
        j=j-1
    return res

def inverse(i, k):
    """
    conversion base 10 à base 4
    """

    reste = 0.0
    diviseur = 4.0 #base 4 dans notre projet
    quotient = 0.0
    dividende = i
    tmp = []
    resultat = []
    
    while dividende != 0:
        reste = dividende % diviseur
        quotient = dividende // diviseur
        dividende = quotient
        tmp.append(str(reste))

    if len(tmp) != k:
        for i in range(k-len(tmp)):
            tmp.append("0")

    for chaine in tmp:
        resultat.append(int(chaine[0]))
    resultat_decode = decode_sequence(resultat)[::-1]
    return resultat[::-1]

def compte_occurences_chevauchantes(sequence, k):
    """
    #sequence -> string
    #k -> longueur du découpage des sous séquences
    #retourne un dictionnaire de sequences de taille k chevauchantes
    """

    res_lettres_occurences = {}
    iteration = 0
    while True:
        substring = sequence[iteration:iteration+k]
        if len(substring) != k:
            break
        if substring not in res_lettres_occurences.keys():
            res_lettres_occurences[substring] = 1
        else:
            res_lettres_occurences[substring] += 1
        iteration += 1
    return res_lettres_occurences

def compte_occurences_chevauchantes2(m, k): #Retourne un tableau conteant à l'indice i l'occurence du mot i.
    res = [0]*(4**k)
    for i in range(len(m)-(k-1)):
        indice = code(m[i:i+k],k)
        res[indice]= res[indice]+1
    return res

def comptage_attendu(nucleotide_frequency, k , longueur_genome):
    indW_occ = {}
    for i in range(4**k-1):
        word = inverse(i,k)
        logproba_word = logproba(word, nucleotide_frequency)
        proba_word = math.exp(logproba_word)
        nb_tmp_occ = 0
        for j in range(k):

            #nb_occ pour un génome de longueur (l - j) -> pour les occurences chevauchantes
            #lorsque j = 0 , génome normal , sinon occurence chevauchante pour des mots de longueur k
            nb_tmp_occ += math.ceil(proba_word * ((longueur_genome-j)/k))

        indW_occ[decode_sequence(word)] = nb_tmp_occ

    return indW_occ

def comptage_attendu2(tuple_frequences, k, l): #Retourne un tableau contenant a l'indice i, l'occurence attendu pour le mot i.
    res = [0]*(4**k)
    for i in range (4**k):
        for d in inverse(i, k):
            if res[i] == 0:
                res[i] = res[i] + tuple_frequences[d]
            else:
                res[i] = res[i] * tuple_frequences[d]
            
    for j in range (4**k):
        res[j] = res[j] * (l-(k-1))
    return res

def simule_sequence(lg, m):

    seq = np.random.choice(4, lg, m) #prend un int entre 0 et 4 de longueur lg et avec la distribution m
    return seq

def nucleotide_frequency_genome(fasta_filepath): #Lit un fichier et nous retourne la fréquence de chaque nucléotide, ici nous concatenons tous les genes.
    dictSequence = read_fasta(fasta_filepath)
    freqA_tot = 0
    freqC_tot = 0
    freqG_tot = 0
    freqT_tot = 0
    nbSeq = 0
    for i in dictSequence.keys():
        (freqA, freqC, freqG, freqT) = nucleotide_frequency(dictSequence[i])
        freqA_tot += freqA
        freqC_tot += freqC
        freqG_tot += freqG
        freqT_tot += freqT
        nbSeq += 1
        
    return (freqA_tot/nbSeq, freqC_tot/nbSeq, freqG_tot/nbSeq, freqT_tot/nbSeq)

#Tracé graphes
def graphe_Attendu_Observe(fasta_filepath, k): #Trace un graph avec pour abscisse les comptage attendu et en ordonnée les comptage observé.
    dictSequence = read_fasta(fasta_filepath)
    sequence_concatene = []
    for i in dictSequence.keys():
        sequence_concatene = sequence_concatene + dictSequence[i]
    
    l = len(sequence_concatene)
    tuple_frequence = nucleotide_frequency_genome(fasta_filepath)
    
    comptage_obs = compte_occurences_chevauchantes2(sequence_concatene, k)
    comptage_att = comptage_attendu2(tuple_frequence,k,l)
    plt.plot(comptage_att, comptage_obs, ".")

    m = (max(tuple_frequence)**k)*l
    plt.plot((range (int (m))))
    plt.ylabel('Observé')
    plt.xlabel('Attendu')
    plt.show


#Tracé graphe
def comparaison_simulation(tuple_frequences, l, k): #Génère 1000 séquence et affiche un graph de points représentant l'occurence moyenne de chaque mot
    c_a = comptage_attendu2(tuple_frequences,k,l)
    esperance_o=[0]*(4**k)
    for i in range(1000):
        res = simule_sequence(l, tuple_frequences)
        c_o = compte_occurences_chevauchantes2(res, k)
        for y in range(4**k):
            esperance_o[y] += c_o[y]
#        plt.plot(c_a, c_o, ".")
    
    for y in range(4**k):
        esperance_o[y] /= 1000
    m = (max(tuple_frequences)**k)*l
    plt.plot(c_a,esperance_o, ".")
    
#    plt.plot((range (int (m))))
    plt.ylabel('Observé')
    plt.xlabel('Attendu')
    plt.show

def proba_mot_n(n, l, tuple_frequences): #Estime la probabilité d'observer un mot un certain nombre n de fois dans une séquence de longueur k
    #Puis en calcul la probabilité empirique    
    comptage = [0, 0, 0, 0]
    mot1 = code([0,3,1,3,2,1], 6) #ATCTGC
    mot2 = code([0,3,0,3,0,3], 6) #ATATAT
    mot3 = code([3,3,3,0,0,0], 6) #TTTAAA
    mot4 = code([0,0,0,0,0,0], 6) #AAAAAA
    
    for i in range(1000):
        seq = simule_sequence(l, tuple_frequences)
        c_o = compte_occurences_chevauchantes2(seq, 6)
        if c_o[mot1] >= n:
            comptage[0] += 1
        if c_o[mot2] >= n:
            comptage[1] += 1
        if c_o[mot3] >= n:
            comptage[2] += 1
        if c_o[mot4] >= n:
            comptage[3] += 1
    for i in range(4):
        comptage[i] /= 1000
    return comptage

#tracé histogramme
def distribution_comptage_mot(mot ,l, tuple_frequences): #Compare la distribution de probabilité avec l'histogramme de la probabilité empirique
    mot_ind = code(mot, len(mot))
    comptage=[0]*(1000)
    for i in range(1000):
        seq = simule_sequence(l, tuple_frequences)
        c_o = compte_occurences_chevauchantes2(seq, len(mot))
        comptage[i]=c_o[mot_ind] #on met les occurences du mot cherché dans comptage[i]
    plt.hist(comptage)
    plt.show()



#PARTIE 3

def estimM(k,tuple_frequences):
    lg = 800
    seq_simule = simule_sequence(lg,tuple_frequences)
    comptage = compte_occurences_chevauchantes2(seq_simule, k)
    M = np.zeros(shape=(4,4))
    for i in range(4**k):
        mot_indice = inverse(i,k)
        proba_mot_indice = comptage[i] / lg
        M[mot_indice[1]][mot_indice[0]] = proba_mot_indice
    return M

#PARTIE 4 Probabilité de mots

def combin(n, k):
    """Nombre de combinaisons de n objets pris k a k"""
    if k > n//2:
        k = n-k
    x = 1
    y = 1
    i = n-k+1
    while i <= n:
        x = (x*i)//y
        y += 1
        i += 1
    return x
    
def fact(n):
    """fact(n): calcule la factorielle de n (entier >= 0)"""
    x=1
    for i in range(2,n+1):
        x*=i
    return x

def proba_occ_comptage_Binomiale(list_compt, k, l, nw):
    """Calcul la probabilité d'occurence avec la formule analytique"""
    res = [0]*(4**k)
    for i in range(4**k):
        p = list_compt[i]/(l-k+1)
        q = 1-p
        for j in range(nw):
            res[i] += combin(l-k+1, j) * math.pow(p,j) * math.pow(q,(l-k+1)-j)
        res[i] = 1 - res[i]
    return res
    
def boucle_n_Binomiale(n,l,tuple_frequence, k, nw):
    """Applique la fonction précédente pour avoir des résultats moyens"""
    res = [0]*(4**k)
    for i in range(n):
        seq = simule_sequence(l, tuple_frequence);
        r = proba_occ_comptage_Binomiale(compte_occurences_chevauchantes2(seq, k),k, l,nw)
        for j in range(4**k):
            res[j]+=r[j]
    for j in range(4**k):
            res[j] /= n
    return res[code([0,0,0,0,0,0],k)]
    
#print(boucle_n_Binomiale(1000,3000,(0.3,0.2,0.2,0.3), 6,1))

    
def proba_occ_comptage_Binomiale_SCIPY(list_compt, k, l, nw):
    """Nous avons utilise scipy car lorsque nw (P(N >= nw))est tres grand, le calcul manuel
    ne fonctionnait plus"""
    res = [0]*(4**k)
    n = l-k+1
    for i in range(4**k):
        p = list_compt[i]/n
        res[i] = 1 - math.fsum(sts.binom.pmf(range(nw), n, p))
    return res
    

def boucle_n_Binomiale_SCIPY(n,l,tuple_frequence, k, nw):
    res = [0]*(4**k)
    for i in range(n):
        seq = simule_sequence(l, tuple_frequence);
        r = proba_occ_comptage_Binomiale_SCIPY(compte_occurences_chevauchantes2(seq, k),k, l,nw)
        for j in range(4**k):
            res[j]+=r[j]
    for j in range(4**k):
            res[j] /= n
    return res[code([0,0,0,1],k)]
    
#print(boucle_n_Binomiale_SCIPY(100,3000,(0.3,0.2,0.2,0.3), 6,1))

def proba_occ_comptage_Poisson(list_compt, k, l, nw):
    """On utilise ici la loi de poisson pour calculer les probabilité d'occurence"""
    res = [0]*(4**k)
    for i in range(4**k):
        p = list_compt[i]
        for j in range(nw):
            res[i] += (math.pow(p,j)* math.exp(-p))/math.factorial(j)
        res[i] = 1 - res[i]
    return res
    
def boucle_n_Poisson(n,l,tuple_frequence, k, nw):
    res = [0]*(4**k)
    for i in range(n):
        seq = simule_sequence(l, tuple_frequence);
        r = proba_occ_comptage_Poisson(occurence_mot(seq, k),k, l,nw)
        for j in range(4**k):
            res[j]+=r[j]
    for j in range(4**k):
            res[j] /= n
    return res[code([0,0,0,1],k)]
    
#print(boucle_n_Poisson(1000,3000,(0.3,0.2,0.2,0.3), 6, 1))




