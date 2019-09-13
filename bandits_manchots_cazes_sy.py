import random
import matplotlib.pyplot as plt
import copy
import math
import pandas as pd
import seaborn as sns
import numpy as np

nbLevier = 4
proba_machine = [random.random() for _ in range(nbLevier)]
proba_estimee = [0 for _ in range(nbLevier)]
coups_effectues = copy.copy(proba_estimee)
epsilon = 0.7
gain_maximal_espere = 0

def gain_binaire(proba_machine, levier_choisi):
    """ float list * int -> int
            hypothese: proba_machine liste de proba float >0
                               levier_choisi [0,nbLevier[
    """
    if random.random() < proba_machine[levier_choisi]:
        return 1
    else:
        return 0


def gain_max(proba_machine, nbCoups):
    """
            list float * int -> int
            hypothese: proba_machine liste de proba float >0
                               nbCoups > 0 nb de coups joues
    """
    global gain_maximal_espere
    levier = proba_machine.index(max(proba_machine))
    gain_maximal_espere += gain_binaire(proba_machine, levier)
    return gain_maximal_espere


def regret(proba_machine, coups_effectues, tab_score):
    """
            float list * int list * int list -> int
            hypothese: proba_machine liste de proba float >0
                               coups_effectues liste recensant le nombre
                               de fois qu'un levier i (coups_effectue[i]
                               est joue
                               tab_score recense les gains pour chacun des
                               leviers (tab_score[i] correspond à la
                               somme des gains)
    """
    g = gain_max(proba_machine, sum(coups_effectues))
    regret = g - sum(tab_score)
    return regret


def algo_random(recompense_moyenne, coups_effectues):
    """
            float list * int list -> int
            hypothese: recompense_moyenne liste de proba float >0
                               coups_effectues liste recensant le nombre
                               de fois qu'un levier i (coups_effectue[i]
                               est joue
    """
    levier = random.randint(0, nbLevier - 1)
    return levier


def algo_greedy(recompense_moyenne, coups_effectues):
    """
        float list * int list -> int
        hypothese: recompense_moyenne liste de proba float >0
                                   coups_effectues liste recensant le nombre
                                   de fois qu'un levier i (coups_effectues[i]
                                   est joue
    """
    levier = recompense_moyenne.index(max(recompense_moyenne))
    return levier


def eps_greedy(recompense_moyenne, coups_effectues):
    """
    float list * int list -> int * int
    hypothese: recompense_moyenne liste de proba float >0
                               coups_effectues liste recensant le nombre
                               de fois qu'un levier i (coups_effectues[i]
                               est joue
                               retourne un entier indiquant si on a une exploitation
                               ou une exploration pour le plot du regret
                               """
    global epsilon
    if random.random() > epsilon:
        return algo_greedy(recompense_moyenne, coups_effectues), 1
    else:
        return algo_random(recompense_moyenne, coups_effectues), 0


def algo_UCB(recompense_moyenne, coups_effectues):
    """
    float list * int list -> int 
    hypothese: recompense_moyenne liste de proba float >0
                               coups_effectues liste recensant le nombre
                               de fois qu'un levier i (coups_effectues[i]
                               est joue
                               """
    evaluation = copy.copy(recompense_moyenne)
    for i in range(len(recompense_moyenne)):
        if recompense_moyenne[i] == 0:
            continue
        evaluation[
            i] += math.sqrt((2 * math.log(sum(coups_effectues))) / coups_effectues[i])
    levier = evaluation.index(max(evaluation))
    return levier


def main(nbCoups, nbExploration=0, n_algo=0):
    """
            int * int * float * int -> void
            hypothese: nbCoups nombre de coups a jouer >0
                               nbExploration nombre de coups dediees a
                               l'exploration nbCoups > 0 et nbExploration <= nbCoups
                               n_algo numero de l'algo a appliquer
                               n_algo = 0 random , 1 pour greedy , 2 eps_greedy , 3 UCB
    """
    global nbLevier
    global proba_machine
    global proba_estimee
    global coups_effectues
    tab_score = [0 for _ in range(nbLevier)]

    # pour plot les regrets
    x_values = np.array([i for i in range(nbCoups)]).flatten()
    exploration_vs_exploitation = []
    list_regret_iteration = []
    y_values = []

    if n_algo == 0:
        for i in range(nbCoups):
            levier = algo_random(proba_estimee, coups_effectues)
            # MAJ des scores
            gain = gain_binaire(proba_machine, levier)
            coups_effectues[levier] += 1
            tab_score[levier] += gain
            proba_estimee[levier] = (
                1 / coups_effectues[levier]) * tab_score[levier]
            list_regret_iteration.append(
                regret(proba_machine, coups_effectues, tab_score))
            exploration_vs_exploitation.append(0)

    if n_algo == 1:
        for i in range(nbExploration):
            levier = algo_random(proba_estimee, coups_effectues)
            gain = gain_binaire(proba_machine, levier)
            coups_effectues[levier] += 1
            tab_score[levier] += gain
            proba_estimee[levier] = (
                1 / coups_effectues[levier]) * tab_score[levier]
            list_regret_iteration.append(
                regret(proba_machine, coups_effectues, tab_score))
            exploration_vs_exploitation.append(0)

        nbExploitation = nbCoups - nbExploration

        for i in range(nbExploitation):
            levier = algo_greedy(proba_estimee, coups_effectues)
            gain = gain_binaire(proba_machine, levier)
            coups_effectues[levier] += 1
            tab_score[levier] += gain
            proba_estimee[levier] = (
                1 / coups_effectues[levier]) * tab_score[levier]
            list_regret_iteration.append(
                regret(proba_machine, coups_effectues, tab_score))
            exploration_vs_exploitation.append(1)

    if n_algo == 2:
        for i in range(nbCoups):
            levier, typeCoup = eps_greedy(proba_estimee, coups_effectues)
            gain = gain_binaire(proba_machine, levier)
            coups_effectues[levier] += 1
            tab_score[levier] += gain
            proba_estimee[levier] = (
                1 / coups_effectues[levier]) * tab_score[levier]
            list_regret_iteration.append(
                regret(proba_machine, coups_effectues, tab_score))
            exploration_vs_exploitation.append(typeCoup)

    if n_algo == 3:
        for i in range(nbExploration):
            levier = algo_random(proba_estimee, coups_effectues)
            print("Exploration",levier)
            gain = gain_binaire(proba_machine, levier)
            coups_effectues[levier] += 1
            tab_score[levier] += gain
            proba_estimee[levier] = (
                1 / coups_effectues[levier]) * tab_score[levier]
            list_regret_iteration.append(
                regret(proba_machine, coups_effectues, tab_score))
            exploration_vs_exploitation.append(0)

        nbExploitation = nbCoups - nbExploration

        for i in range(nbExploitation):
            levier = algo_UCB(proba_estimee, coups_effectues)
            print("Exploitation UCB: ", levier)
            gain = gain_binaire(proba_machine, levier)
            coups_effectues[levier] += 1
            tab_score[levier] += gain
            proba_estimee[levier] = (
                1 / coups_effectues[levier]) * tab_score[levier]
            list_regret_iteration.append(
                regret(proba_machine, coups_effectues, tab_score))
            exploration_vs_exploitation.append(1)

    # Plot des regrets (on distingue exploration et exploitation)
    if n_algo == 2:
        y_values = np.array(copy.copy(list_regret_iteration)).flatten()
        y_0 = [0 for _ in range(nbCoups)]
        x_0 = [0 for _ in range(nbCoups)]

        for i in range(nbCoups):
            if exploration_vs_exploitation[i] == 0:
                x_0[i] = i
                y_0[i] = y_values[i]

        d = {"x": x_values, "y": y_values,
             "col": exploration_vs_exploitation, "x_0": x_0, "y_0": y_0}
        df = pd.DataFrame(d)
        sns.lineplot(x="x", y="y", data=df)
        sns.scatterplot(x="x_0", y="y_0", color='r', data=df)
        plt.xlabel("nbCoups")
        plt.ylabel("Regret")
        plt.title("Regret de l'algorithme epsilon greedy")
        plt.show()

    else:
        y_values = np.array(copy.copy(list_regret_iteration)).flatten()
        d = {"x": x_values, "y": y_values, "col": exploration_vs_exploitation}
        df = pd.DataFrame(d)
        sns.lineplot(x="x", y="y", hue="col", data=df)
        plt.xlabel("nbCoups")
        plt.ylabel("Regret")
        if n_algo == 0:
            plt.title("Regret de l'algorithme aléatoire")
        if n_algo == 1:
            plt.title("Regret de l'algorithme greedy")
        if n_algo == 3:
            plt.title("Regret de l'algorithme UCB")
        plt.show()

# main(nbCoups, nbExploration, n_algo)
#main(100)  # algo random test 100 coups
#main(100,75,1) #algo greedy 100 coups 25 exploration
#main(100, 0, 2)  # eps_greedy
main(100,50,3) #UCB
