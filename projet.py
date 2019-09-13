import math
import utils as ut
from functools import reduce
import operator
import collections.abc
from scipy.stats import chi2_contingency
import pandas as pd
import matplotlib.pyplot as plt

# CAZES Hadrien SY Virak Bot Baramy


def map_nested_dicts(ob, func):
    if isinstance(ob, collections.abc.Mapping):
        return {k: map_nested_dicts(v, func) for k, v in ob.items()}
    else:
        return func(ob)


def getPrior(dataset):
    res = dict()
    estimation = dataset['target'].sum() / dataset['target'].size
    res['estimation'] = estimation
    variance = estimation * (1 - estimation)
    min5pourcent = estimation - 1.96 * math.sqrt(variance/dataset['target'].size)
    max5pourcent = estimation + 1.96 * math.sqrt(variance/dataset['target'].size)
    res['min5pourcent'] = min5pourcent
    res['max5pourcent'] = max5pourcent
    return res
            


class APrioriClassifier(ut.AbstractClassifier):

    def __init__(self):
        pass

    def estimClass(self, attrs):
        return 1

    def statsOnDF(self, df):
        """
        res = {'VP': 0, 'VN': 0, 'FP': 0, 'FN': 0 , 'Precision': 0 , 'Rappel': 0 }
        classMajoritaire = self.estimClass(df)
        for t in df.itertuples():
            dic = t._asdict()
            v = dic['target']
            if v == 1 and classMajoritaire == 1:
                res['VP'] += 1
            elif v == 0 and classMajoritaire == 0:
                res['VN'] += 1
            elif v == 0 and classMajoritaire == 1:
                res['FP'] += 1
            else:
                res['FN'] += 1

        res['Précision'] = res['VP'] / (res['VP'] + res['FP'])
        res['Rappel'] = res['VP'] / (res['VP'] + res['FN'])
        return res
        
        """
        res = {'VP': 0, 'VN': 0, 'FP': 0, 'FN': 0 , 'Precision': 0 , 'Rappel': 0 }
        
        i = 0
        for t in df.itertuples():
            dic = t._asdict()
            v = dic['target']
            attrs = ut.getNthDict(df,i)
            classePrevue = self.estimClass(attrs)

            if v == 1 and classePrevue == 1:
                res['VP'] += 1
            elif v == 0 and classePrevue == 0:
                res['VN'] += 1
            elif v == 0 and classePrevue == 1:
                res['FP'] += 1
            else:
                res['FN'] += 1
            i += 1

        res['Precision'] = res['VP'] / (res['VP'] + res['FP'])
        res['Rappel'] = res['VP'] / (res['VP'] + res['FN'])
        return res

                
def P2D_l(df,attr):
    """
    res = dict()
    value = df.groupby('target')[attr].value_counts() / df.groupby('target')[attr].count()
    for i in df.target.unique():
        tmp = dict()
        for j in df[attr].unique():
            tmp[j] = value[i][j]
        res[i] = tmp
    return res
    """
    res = dict()
    # initialisation des valeurs du dictionnaire
    for target in df['target'].unique():
        tmp = dict()
        for val_attr in df[attr].unique():
            tmp[val_attr] = 0
        res[target] = tmp

    size_of_df = df.groupby('target')[attr].count()  # nb de classe 0 et de classe 1

    # maj des valeurs du dictionnaire
    for t in df.itertuples():
        dictio = t._asdict()
        target = dictio['target'] #prend la valeur de target du nuplet
        attribut = dictio[attr] #prend la valeur de l'attribut du nuplet
        res[target][attribut] += 1
    
    # calcul de la probabilité
    for target in res.keys():
        for val_attribut in res[target].keys():
            res[target][val_attribut] /= size_of_df[target]

    return res

def P2D_p(df,attr):
    res = dict()
    # initialisation des valeurs du dictionnaire
    for val_attr in df[attr].unique():
        tmp = dict()
        for target in df['target'].unique():
            tmp[target] = 0
        res[val_attr] = tmp
    
    size_of_df = df.groupby(attr)['target'].count()  # nb d'elements de classe de l'attribut

    # maj des valeurs du dictionnaire
    for t in df.itertuples():
        dictio = t._asdict()
        target = dictio['target'] #prend la valeur de target du nuplet
        attribut = dictio[attr] #prend la valeur de l'attribut du nuplet
        res[attribut][target] += 1
    

    # calcul de la probabilité
    for val_attribut in df[attr].unique():
        for target in df['target'].unique():
            res[val_attribut][target] /= size_of_df[val_attribut]

    return res

class ML2DClassifier(APrioriClassifier):

    def __init__(self, df, attr):
        self.df = df
        self.attr = attr
        self.P2D_l = P2D_l(df,self.attr)

    def estimClass(self, attrs):
        val_attr = attrs[self.attr]
        list_proba = []
        list_key = list(self.P2D_l.keys()) # [1,0]
        list_key.reverse()
        for i in list_key:
            list_proba.append(self.P2D_l[i][val_attr])

        target = list_proba.index(max(list_proba))
        return target
            

class MAP2DClassifier(APrioriClassifier):

    def __init__(self, df, attr):
        self.df = df
        self.attr = attr
        self.P2D_p = P2D_p(df,self.attr)

    def estimClass(self, attrs):
        val_attr = attrs[self.attr]
        list_proba = []
        list_key = list(self.P2D_p[val_attr].keys()) # [0,1,2,3] pour thal
        list_key.reverse()
        for i in list_key:
            list_proba.append(self.P2D_p[val_attr][i])

        target = list_proba.index(max(list_proba))
        return target
            
def nbParams(df,list_attr=None):
    TAILLE_1_VAL = 8 # 8 octets
    taille_totale = 0
    list_nb_val_attribut = []

    if list_attr == None:
        list_attr = list(df)

    for attribut in list_attr:
        list_nb_val_attribut.append(len(df[attribut].unique()))

    nb_valeurs_attribut_total = reduce(operator.mul, list_nb_val_attribut, 1)         
    taille_totale = nb_valeurs_attribut_total * TAILLE_1_VAL

    nb_variables = len(list_attr)

    print ("{} variable(s) : {} octets".format(nb_variables , taille_totale))

def nbParamsIndep(df):
    TAILLE_1_VAL = 8
    taille_totale = 0
    list_attr = list(df)

    for attribut in list_attr:
        nb_valeurs_attribut = len(df[attribut].unique())
        taille_totale += nb_valeurs_attribut * TAILLE_1_VAL

    nb_variables = len(list_attr)

    print ("{} variable(s) : {} octets".format(nb_variables , taille_totale))
    

def drawNaiveBayes(df,nom_attribut_classe):
    chaine_draw = nom_attribut_classe
    list_attr = list(df)
    list_attr.remove(nom_attribut_classe)
    for attribut in list_attr:
        chaine_draw += "->" + attribut + ";"
        chaine_draw += nom_attribut_classe
    return ut.drawGraph(chaine_draw)

def nbParamsNaiveBayes(df,nom_attribut_classe,list_attr = None):
    TAILLE_1_VAL = 8
    taille_totale = 0

    # si la liste des attributs est vide , la taille totale de la table se fera uniquement selon 'target'
    if list_attr == []: 
        nb_variables = 0
        taille_totale = len(df[nom_attribut_classe].unique()) * TAILLE_1_VAL

    # si la list des attributs est vide , on le fait selon tout les attributs
    else:
        if list_attr == None:
            list_attr = list(df)

        nb_variables = len(list_attr)
        list_nb_val_attribut = []

        for attribut in list_attr:
            list_nb_val_attribut.append(len(df[attribut].unique()))

        list_taille_table = [nb_val_attribut * len( df[nom_attribut_classe].unique() ) * TAILLE_1_VAL for nb_val_attribut in list_nb_val_attribut]

        taille_totale = sum(list_taille_table) - ( len(df[nom_attribut_classe].unique()) * TAILLE_1_VAL ) # car P(target|target) = P(target)

    print ("{} variable(s) : {} octets".format(nb_variables , taille_totale))

class MLNaiveBayesClassifier(APrioriClassifier):
    """
    Naive Bayes avec maximum de vraisemblance
    """
    def __init__(self,df):

        self.df = df

        self.list_attr = list(df)
        self.list_attr.remove('target')

        self.dict_key_unique_values = dict()

        for attribut in self.list_attr:
            self.dict_key_unique_values[attribut] = list(self.df[attribut].unique())

        self.list_table_cond = []
        for attribut in self.list_attr:
            self.list_table_cond.append(P2D_l(self.df, attribut))


    def estimProbas(self,attrs):

        vals_target = list(self.df['target'].unique())
        vals_target.reverse()
        vraisemblance = {}

        #initialisation des targets à 0 

        for target in vals_target:
            vraisemblance[target] = 0
            list_proba_tmp = []
            i = 0
            #maj des vraisemblances des target
            for attribut_i in self.list_attr:
                
                val_attr = attrs[attribut_i]
                dic_tmp = self.list_table_cond[i]
                if val_attr not in self.dict_key_unique_values[attribut_i]:
                    proba_attribut_i = 0.0
                else:
                    proba_attribut_i = dic_tmp[target][val_attr]

                list_proba_tmp.append(proba_attribut_i)
                i += 1
            #print(list_proba_tmp , target)
            #list(filter(lambda v: v != 0.0, list_proba_tmp)) # on filtre les 0 car ils faussent le calcul de vraisemblance 
            vraisemblance[target] = reduce(operator.mul, list_proba_tmp, 1)  
        return vraisemblance

    def estimClass(self,attrs):

        vraisemblance = self.estimProbas(attrs)
        return max(vraisemblance, key=lambda t: vraisemblance[t])




class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    Naive Bayes avec maximum a posteriori
    """

    def __init__(self,df):

        self.df = df
        self.list_attr = list(df)
        self.list_attr.remove('target')

        self.dict_key_unique_values = dict()

        for attribut in self.list_attr:
            self.dict_key_unique_values[attribut] = list(self.df[attribut].unique())

        self.list_table_cond = []
        for attribut in self.list_attr:
            self.list_table_cond.append(P2D_l(self.df, attribut))


    def estimProbas(self,attrs):

        vraisemblance = {}
        vals_target = list(self.df['target'].unique())
        vals_target.reverse()

        #initialisation des targets à 0 

        for target in vals_target:
            vraisemblance[target] = 0
            list_proba_tmp = []
            i = 0
            #maj des vraisemblances des target
            for attribut_i in self.list_attr:
                
                val_attr = attrs[attribut_i]
                dic_tmp = self.list_table_cond[i]
                if val_attr not in self.dict_key_unique_values[attribut_i]:
                    proba_attribut_i = 0.0
                else:
                    proba_attribut_i = dic_tmp[target][val_attr]

                list_proba_tmp.append(proba_attribut_i)
                i += 1
            
            vraisemblance[target] = reduce(operator.mul, list_proba_tmp, 1)

        aPosteriori = dict()
        p_attr_i_a_n = 1

        # On determine P(target)
        p_target = self.df.target.value_counts() / self.df.target.count()

        evidence = vraisemblance[0] * p_target[0] + vraisemblance[1] * p_target[1]

        for target in vals_target:
            aPosteriori[target] = (vraisemblance[target] * p_target[target]) / evidence

        return aPosteriori
        

    def estimClass(self,attrs):

        aPosteriori = self.estimProbas(attrs)
        return max(aPosteriori, key=lambda t: aPosteriori[t])

        
def isIndepFromTarget(df,attr,x):
    contingence = pd.crosstab(df[attr],df.target).values #conversion d'un dataframe en numpy array avec values
    g, p, dof, expctd = chi2_contingency(contingence)
    if p < x:
        return False
    return True

class ReducedMLNaiveBayesClassifier(APrioriClassifier):
    
    def __init__(self, df, seuil):
        self.df = df
        self.seuil = seuil

        self.list_attr = list(df)
        self.list_attr.remove('target') # on ne considere pas target

        for attr in self.list_attr: # si l'attribut est indépendant au seuil donne , on ne le considere pas
            attribut_dependance = isIndepFromTarget(self.df, attr, self.seuil)
            if attribut_dependance == True:
                self.list_attr.remove(attr)

        self.dict_key_unique_values = dict()

        for attribut in self.list_attr:
            self.dict_key_unique_values[attribut] = list(self.df[attribut].unique())

        self.list_table_cond = []
        for attribut in self.list_attr:
            self.list_table_cond.append(P2D_l(self.df, attribut))

    def estimProbas(self, attrs):

        vals_target = list(self.df['target'].unique())
        vals_target.reverse()
        vraisemblance = {}

        #initialisation des targets à 0 

        for target in vals_target:
            vraisemblance[target] = 0
            list_proba_tmp = []
            i = 0
            #maj des vraisemblances des target
            for attribut_i in self.list_attr:
                
                val_attr = attrs[attribut_i]
                dic_tmp = self.list_table_cond[i]
                if val_attr not in self.dict_key_unique_values[attribut_i]:
                    proba_attribut_i = 0.0
                else:
                    proba_attribut_i = dic_tmp[target][val_attr]

                list_proba_tmp.append(proba_attribut_i)
                i += 1
            #print(list_proba_tmp , target)
            #list(filter(lambda v: v != 0.0, list_proba_tmp)) # on filtre les 0 car ils faussent le calcul de vraisemblance 
            vraisemblance[target] = reduce(operator.mul, list_proba_tmp, 1)  
        return vraisemblance

    def estimClass(self,attrs):

        vraisemblance = self.estimProbas(attrs)
        return max(vraisemblance, key=lambda t: vraisemblance[t])

    def draw(self):
        chaine_draw = 'target'
        for attribut in self.list_attr:
            chaine_draw += "->" + attribut + ";"
            chaine_draw += 'target'
        return ut.drawGraph(chaine_draw)

class ReducedMAPNaiveBayesClassifier(APrioriClassifier):

    def __init__(self, df, seuil):
        self.df = df
        self.seuil = seuil

        self.list_attr = list(df)
        self.list_attr.remove('target') # on ne considere pas target

        for attr in self.list_attr: # si l'attribut est indépendant au seuil donne , on ne le considere pas
            attribut_dependance = isIndepFromTarget(self.df, attr, self.seuil)
            if attribut_dependance == True:
                self.list_attr.remove(attr)

        self.dict_key_unique_values = dict()

        for attribut in self.list_attr:
            self.dict_key_unique_values[attribut] = list(self.df[attribut].unique())

        self.list_table_cond = []
        for attribut in self.list_attr:
            self.list_table_cond.append(P2D_l(self.df, attribut))

    def estimProbas(self,attrs):

        vraisemblance = {}
        vals_target = list(self.df['target'].unique())
        vals_target.reverse()

        #initialisation des targets à 0 

        for target in vals_target:
            vraisemblance[target] = 0
            list_proba_tmp = []
            i = 0
            #maj des vraisemblances des target
            for attribut_i in self.list_attr:
                
                val_attr = attrs[attribut_i]
                dic_tmp = self.list_table_cond[i]
                if val_attr not in self.dict_key_unique_values[attribut_i]:
                    proba_attribut_i = 0.0
                else:
                    proba_attribut_i = dic_tmp[target][val_attr]

                list_proba_tmp.append(proba_attribut_i)
                i += 1
            
            vraisemblance[target] = reduce(operator.mul, list_proba_tmp, 1)

        aPosteriori = dict()
        p_attr_i_a_n = 1

        # On determine P(target)
        p_target = self.df.target.value_counts() / self.df.target.count()

        evidence = vraisemblance[0] * p_target[0] + vraisemblance[1] * p_target[1]

        for target in vals_target:
            aPosteriori[target] = (vraisemblance[target] * p_target[target]) / evidence

        return aPosteriori
        

    def estimClass(self,attrs):

        aPosteriori = self.estimProbas(attrs)
        return max(aPosteriori, key=lambda t: aPosteriori[t])

    def draw(self):
        chaine_draw = 'target'
        for attribut in self.list_attr:
            chaine_draw += "->" + attribut + ";"
            chaine_draw += 'target'
        return ut.drawGraph(chaine_draw)

def mapClassifiers(dic,df):
    """
    pour dic: 
    clé du dictionnaire 1->7
    valeur d'une clé dans le dictionnaire : un classifieur
    df = dataframe
    """

    precision_list = [] #coordonnee précision 
    rappel_list = [] #coordonnee rappel
    annotation_list = [i for i in range(1,len(dic.keys())+1 )]

    for classifieur in dic.values():
        res_tmp = classifieur.statsOnDF(df)
        precision_list.append(res_tmp['Precision'])
        rappel_list.append(res_tmp['Rappel'])

    fig, ax = plt.subplots()
    ax.scatter(precision_list, rappel_list, marker="x", c = "r")

    for i in range( len( dic.keys() )):
        ax.annotate(annotation_list[i] , (precision_list[i], rappel_list[i]))
    plt.show()

def MutualInformation(df,X,Y):
    proba_table = df.groupby(X)[Y].value_counts() / df.groupby(X)[Y].count()

    information = 0.0

    list_values_index = proba_table.index.values.tolist()
    dict_key_unique_values = {}

    for x in df[X].unique():
        dict_key_unique_values[x] = []

    for (x,y) in list_values_index:
        dict_key_unique_values[x].append(y)

    for x in df[X].unique():

        P_x = (df[X].value_counts().div(len(df)))[x]

        for y in df[Y].unique():

            if y not in dict_key_unique_values[x]:
                continue

            P_y = (df[Y].value_counts().div(len(df)))[y]
            Px_y = proba_table[x][y] * P_x # ou P_y
            
            information += Px_y * math.log(Px_y/(P_x * P_y) ,2)

    return information

def ConditionalMutualInformation(df,X,Y,Z):

    mutualInformation = 0.0

    proba_table_Z_Y = df.groupby(Z)[Y].value_counts() / df.groupby(Z)[Y].count()
    proba_table_Y_X = df.groupby(Y)[X].value_counts() / df.groupby(Y)[X].count()

    dict_key_unique_values_Z_Y = {}
    dict_key_unique_values_Y_X = {}

    list_values_index_Z_Y = proba_table.index.values.tolist()
    list_values_index_Y_X = proba_table.index.values.tolist()
    

    # Z sera la target
    # on sait que X ou Y est independant de Z = la target , on peut donc ecrire
    # P(X,Y,Z) = P(X) * P(Y|X) * P(Z|Y) ou P(X,Y,Z) = P(Y) * P(X|Y) * P(Z|X)

    for z in df[Z].unique(): 
        dict_key_unique_values_Z_Y[z] = []

    for y in df[Y].unique():
        dict_key_unique_values_Y_X[y] = []

    for (z,y) in list_values_index_Z_Y:
        dict_key_unique_values_Z_Y[z].append(y)

    for (y,x) in list_values_index_Y_X:
        dict_key_unique_values_Y_X[y].append(x)

    for z in df[Z].unique():

        P_z = (df[Z].value_counts().div(len(df)))[z]


        for x in df[X].unique():

            if x not in dict_key_unique_values_Y_X[x]:
                continue

            P_x = (df[X].value_counts().div(len(df)))[x]
            
            for y in df[Y].unique():

                if y not in dict_key_unique_values_Z_Y[y]:
                    continue

                P_y = (df[X].value_counts().div(len(df)))[x]

                # P_y_x = P(Y|X)
                Py_x =  proba_table_Y_X[x][y]
                Pz_y = proba_table_Z_Y[z][y]

                Px_y_z = P_x * Py_x * Pz_y

    return mutualInformation









