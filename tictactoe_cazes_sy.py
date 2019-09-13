import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random
## Constante
OFFSET = 0.2



class State:
    """ Etat generique d'un jeu de plateau. Le plateau est represente par une matrice de taille NX,NY,
    le joueur courant par 1 ou -1. Une case a 0 correspond a une case libre.
    * next(self,coup) : fait jouer le joueur courant le coup.
    * get_actions(self) : renvoie les coups possibles
    * win(self) : rend 1 si le joueur 1 a gagne, -1 si le joueur 2 a gagne, 0 sinon
    * stop(self) : rend vrai si le jeu est fini.
    * fonction de hashage : renvoie un couple (matrice applatie des cases, joueur courant).
    """
    NX,NY = None,None
    def __init__(self,grid=None,courant=None):
        self.grid = copy.deepcopy(grid) if grid is not None else np.zeros((self.NX,self.NY),dtype="int")
        self.courant = courant or 1
    def next(self,coup):
        pass
    def get_actions(self):
        pass
    def win(self):
        pass
    def stop(self):
        pass
    @classmethod
    def fromHash(cls,hash):
        return cls(np.array([int(i)-1 for i in list(hash[0])],dtype="int").reshape((cls.NX,cls.NY)),hash[1])
    def hash(self):
        return ("".join(str(x+1) for x in self.grid.flat),self.courant)
            
class Jeu:
    """ Jeu generique, qui prend un etat initial et deux joueurs.
        run(self,draw,pause): permet de joueur une partie, avec ou sans affichage, avec une pause entre chaque coup. 
                Rend le joueur qui a gagne et log de la partie a la fin.
        replay(self,log): permet de rejouer un log
    """
    def __init__(self,init_state = None,j1=None,j2=None):
        self.joueurs = {1:j1,-1:j2}
        self.state = copy.deepcopy(init_state)
        self.log = None
    def run(self,draw=False,pause=0.5):
        log = []
        if draw:
            self.init_graph()
        while not self.state.stop():
            coup = self.joueurs[self.state.courant].get_action(self.state)
            log.append((self.state,coup))
            self.state = self.state.next(coup)
            if draw:
                self.draw(self.state.courant*-1,coup)
                plt.pause(pause)
        if draw:    
            plt.pause(pause*3)
        return self.state.win(),log
    def init_graph(self):
        self._dx,self._dy  = 1./self.state.NX,1./self.state.NY
        self.fig, self.ax = plt.subplots()
        for i in range(self.state.grid.shape[0]):
            for j in range(self.state.grid.shape[1]):
                self.ax.add_patch(patches.Rectangle((i*self._dx,j*self._dy),self._dx,self._dy,\
                        linewidth=1,fill=False,color="black"))
        plt.show(block=False)
    def draw(self,joueur,coup):
        color = "red" if joueur>0 else "blue"
        self.ax.add_patch(patches.Rectangle(((coup[0]+OFFSET)*self._dx,(coup[1]+OFFSET)*self._dy),\
                        self._dx*(1-2*OFFSET),self._dy*(1-2*OFFSET),linewidth=1,fill=True,color=color))
        plt.draw()
    def replay(self,log,pause=0.5):
        self.init_graph()
        for state,coup in log:
            self.draw(state.courant,coup)
            plt.pause(pause)


class MorpionState(State):
    """ Implementation d'un etat du jeu du Morpion. Grille de 3X3. 
    """
    NX,NY = 3,3
    def __init__(self,grid=None,courant=None):
        super(MorpionState,self).__init__(grid,courant)
    def next(self,coup):
        state =  MorpionState(self.grid,self.courant)
        state.grid[coup]=self.courant
        state.courant *=-1
        return state
    def get_actions(self):
        return list(zip(*np.where(self.grid==0)))
    def win(self):
        for i in [-1,1]:
            if ((i*self.grid.sum(0))).max()==3 or ((i*self.grid.sum(1))).max()==3 or ((i*self.grid)).trace().max()==3 or ((i*np.fliplr(self.grid))).trace().max()==3: return i
        return 0
    def stop(self):
        return self.win()!=0 or (self.grid==0).sum()==0
    def __repr__(self):
        return str(self.hash())

class Agent:
    """ Classe d'agent generique. Necessite une methode get_action qui renvoie l'action correspondant a l'etat du jeu state"""
    def __init__(self):
        pass
    def get_action(self,state):
        pass


class AgentAleatoire(Agent):
    def __init__(self):
        self.name = "Aleatoire"
        self.color = "blue"
    def get_action(self, state):
        actions = state.get_actions()
        index = np.random.randint(0, len(actions))
        return actions[index]
                                  
class AgentMonteCarlo(Agent):
    def __init__(self, N = 10):
        self.N = N
        self.name = "MC N="+str(self.N)
        self.color = "red"

    def get_action(self, state):
        actions = state.get_actions()
        record_win = np.zeros(len(actions), dtype='int')
        record_nb = np.zeros(len(actions), dtype='int')

        j1, j2 = AgentAleatoire(), AgentAleatoire()
        
        for i in range(0, self.N):
            index = np.random.randint(0, len(actions))
            record_nb[index] += 1
            
            new_state = state.next(actions[index])
            m = MorpionState(new_state.grid, new_state.courant)
            jeu = Jeu(m, j1, j2)
            winner = jeu.run()[0]
            
            if winner == state.courant:
                record_win[index] += 1
       
        rewards = np.nan_to_num(record_win / record_nb)
        action = actions[rewards.argmax()]
        # print(state.courant, action)
        # print(actions)
        # print(rewards)
        # print()
        return action

class Node(object):
    def __init__(self,action = None,parentNode=None, state=None , facteur = math.sqrt(2)):
        self.action = action #l'action a jouer
        self.parentNode = parentNode #None par defaut pour la racine
        self.state = state #l'etat du board a un instant t
        self.children = [] #la liste des actions possibles sous forme de noeud !
        self.wins = 0 # permettra d'avoir la probabilite empirique
        self.visits = 0 # permettra d'avoir la probabilite empirique
        self.actions_possibles = state.get_actions() # liste les actions possibles
        self.dernierJoueur = state.courant * -1

    def add_children(self, action , state):
        n = Node(action, parentNode = self , state = state) #permet de creer un noeud dont le parent est celui qui le cree
        self.actions_possibles.remove(action)
        self.children.append(n)
        #print("nouveau noeud: " + str(n.action) + " pere: " + str(self.action))
        return n

    def upd(self,winner):
        self.visits = self.visits + 1
        if winner == self.dernierJoueur:
            self.wins = self.wins + 1


    def infos(self):
        #if self.parentNode != None:
            #print("node wins: " + str(self.wins) + " visited: " + str(self.visits) + " move: " + str(self.action) + " action pere: " + str(self.parentNode.action))
            #s = sorted(self.children, key = lambda n: n.wins/n.visits + math.sqrt(2*math.log(self.visits)/n.visits))
        st = ""
        for i in self.children:
            st += str(i.action) + " "
        print (st)

    def UCTSelect(self, facteur = 1):
        proba_children = []
        coeff_conf = []
        score = []
        for i in self.children:
            if i.actions_possibles == [] and i.state.stop() == True and i.state.win() == i.dernierJoueur:
                return i
            proba_children.append(i.wins/i.visits)
            coeff_conf.append(math.sqrt(2 * math.log(self.visits)/i.visits))
        for (x,y) in zip(proba_children,coeff_conf):
            score.append(x+facteur*y) #default math.sqrt(2) * y
        #print(score)
        chosen = score.index(max(score))
        #print(self.children[chosen])
        return self.children[chosen]

class AgentMCTS(Agent):
    def __init__(self, nbIter = 10, facteur = 1):
        self.nbIter = nbIter
        self.facteur = facteur
        self.name = "MCTS N=" + str(nbIter) + " F=" + str(self.facteur)
        self.color = "green"

    def get_action(self, state):
        racine = Node(state= state) #la racine n'a pas de coup joué donc action = None et la racine n'a pas de pere donc None
        for i in range(0,self.nbIter):
            node = racine
            new_state = copy.copy(state)
            # Pour nbIter = 0 , ne rentre pas dedans car racine n'a pas d'enfants a cet instant.
            # PHASE DE SELECTION : quand on a deja cree tous les noeuds du noeud actuel
            while node.actions_possibles == [] and node.children != []:
                node = node.UCTSelect(self.facteur)
                #print("node actuel:" + str(node.action) + " pere: " + str(node.parentNode.action))
                new_state = new_state.next(node.action)
            
            # PHASE EXPANSION: on parcourt les actions possibles et on va l'explorer
            if node.actions_possibles != [] and new_state.stop() != True and new_state.get_actions() != []:
                action = random.choice(node.actions_possibles) # on teste une action de facon aleatoire
                new_state = new_state.next(action) #l'etat du board change avec le coup qui a ete posee                    
                node = node.add_children(action,new_state) #rajoute un enfant au noeud courant et on continue a le parcourir

            # PHASE SIMULATION:
            while new_state.get_actions() != [] and new_state.stop() != True: # tant que l'etat dans lequel on est n'est pas terminal
                new_state = new_state.next(random.choice(new_state.get_actions()))

            # PHASE UPDATE: On est en bas de l'arbre , on veut remonter le resultat
            while node != None: # tant qu'on n'est pas remonte a la racine
                node.upd(new_state.win())
                #if node.parentNode != None:
                    #print("je remonte node actuel:" + str(node.action) + " pere: " + str(node.parentNode.action))
                #else:
                    #print("je suis à la racine")
                node = node.parentNode

        #s = sorted(racine.children, key = lambda c: c.wins/c.visits)[-1].action
        #print("coup decide = " + str(s))
        proba_children = []
        for i in racine.children:
            proba_children.append(i.wins/i.visits)
        chosen = proba_children.index(max(proba_children))
        return racine.children[chosen].action

def main():
    j1, j2 = AgentMonteCarlo(),AgentMCTS()
    morpion = MorpionState()
    jeu = Jeu(morpion, j1, j2)
    jeu.run(True, pause=1)


#main()

def affiche():
    j1, j2 = AgentMonteCarlo(30), AgentMCTS(30, 0.35)
    
    nbGames = 301
    step = 25
    x_values = []
    j1_wins = [] #j1 wins
    j2_wins = [] #j2 wins
    for i in range(1, nbGames+1, step):
        nbWins1 = 0
        nbWins2 = 0
        for j in range(i):
            morpion = MorpionState()
            jeu = Jeu(morpion, j1, j2)
            winner = jeu.run()[0]

            if winner == 1:
                nbWins1 += 1
            elif winner == -1:
                nbWins2 += 1
                
        j1_wins.append(nbWins1)
        j2_wins.append(nbWins2)

        x_values.append(i)
        print(i)

    y_valuesj1 = [0]
    y_valuesj2 = [0]
    for i in range(1, len(j1_wins)):
        y_valuesj1.append(j1_wins[i] / x_values[i])
        y_valuesj2.append(j2_wins[i] / x_values[i])
        
    print(x_values, y_valuesj1)

    
    fig, ax = plt.subplots()
    ax.plot(x_values, y_valuesj1, label=j1.name, color=j1.color)
    ax.plot(x_values, y_valuesj2, label=j2.name, color=j2.color)
    legend = ax.legend(loc='lower right', shadow=False)
    # plt.plot(x_values, y_valuesj1, label='j1')
    # plt.plot(x_values, y_valuesj2, label='j2')
    plt.xlabel('Nombre de parties')
    plt.ylabel('Proba de victoires')
    plt.title(j1.name + " VS " + j2.name)
    plt.show()
affiche()
