import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

class Graphe:
        def __init__(self,tailleGraphe,p):
                self.N=tailleGraphe
                self.cout= random.randint(0,self.N) # En attendant d'avoir le vrai cout
                self.connexe= False
                self.graphe=np.zeros((self.N,self.N))
                while self.connexe==False:
					for i in range(self.N):
						for j in range(i,self.N):
							if random.random()<p and i != j:
								self.graphe[i,j]=1
								self.graphe[j,i]=1
					self.connexe=self.isConnexe()
					test = self.isConnexe2()
					#pour verifier que les deux methodes isConnexe donnent les memes resultats
					if self.connexe!= test:
						print "ERROR"
								

        def nodesAndEdges(self):
            nodes = []
            edges = []
            for i in range(self.N):
                    nodes.append(i)
                    for j in range(i,self.N):
                        if self.graphe[i,j]==1:
                            edges.append((i,j))
            return (nodes,edges)


        def degrees(self):
                deg={}
                for i in range(self.N):
                        deg[i]=0 #on met toutes les valeurs du dico a 0
                for i in range(self.N):
                        deg[sum(self.graphe[i])]+=1

                return deg
        
        def sceDegrees(self):
                deg = self.degrees()
                print ("deg",deg)

                gamma = 2.5
                th=[0]*self.N
                for i in range(self.N):
                        if deg[i]!=0:
                                th[i] = deg[i]**(-gamma)
                #print (th)
                
                kmin=0
                kmax=self.N-1
                i=0
                while deg[i]==0:
                        kmin+=1
                        i+=1
                i=self.N-1
                while deg[i]==0:
                        kmax-=1
                        i-=1
                        
                sce=0
                for i in range(kmin,kmax):
                        sce+=(th[i]-deg[i])**2
                return sce


        
        def sceCk(self):
                
                Ci=[0]*self.N #stocke les coefficients de clustering pour chaque noeud
                ni=[0]*self.N

                #pour chaque noeuds on garde en memoire ses voisins, grace a un dictionnaire : dicoN
                dicoN = {}
                for i in range(self.N):
                        dicoN[i]=[]
                for i in range(self.N):
                        for j in range(self.N):
                                if self.graphe[i,j]==1:
                                        dicoN[i].append(j)
                print("dicoN:",dicoN)

                #on stocke dans ni le nombres de liens entre les voisins d'un noeuds
                for i in range(self.N):
                        n=0
                        if len(dicoN[i])!=0 and len(dicoN[i])!=1: #si le noeud n'a aucun ou un seul voisin ni vaudra 0 et Ci aussi
                                for j in range(len(dicoN[i])-1): 
                                        for k in range(j+1,len(dicoN[i])):
                                                #print(j,k,dicoN[i],"\n")
                                                if self.graphe[dicoN[i][j],dicoN[i][k]]==1:
                                                        n+=1
                        ni[i]=n

                print ("ni:",ni)
                
                for i in range(self.N):
                        if sum(self.graphe[i])!=0 and sum(self.graphe[i])!=1:
                                Ci[i]=2*ni[i]/(sum(self.graphe[i])*(sum(self.graphe[i])-1))
                #print (Ci)

                Ck={}
                for i in range(self.N+1):
                        Ck[i]=0 
                for i in range(self.N+1):
                        somme = 0
                        k=0
                        for j in range(self.N):
                                if sum(self.graphe[j])==i:
                                        somme=somme+Ci[j]
                                        k+=1
                        if k!=0:
                                Ck[i]= somme/k

                #print (Ck)
                                                            
                deg = self.degrees()
                
                kmin=0
                kmax=self.N-1
                i=0
                while deg[i]==0:
                        kmin+=1
                        i+=1
                i=self.N-1
                while deg[i]==0:
                        kmax-=1
                        i-=1

                th=1/(kmax-kmin)
                sce=0
                for i in range(kmin,kmax+1):
                        sce+=(th-Ck[i])**2
                return sce
                

        def isConnexe(self):

                # Creation de la matrice identite
                matrice =np.eye(self.N)
                # Matrice correspondant a la somme de la matrice identite et de notre graphe
                # On cree une matrice temporaire qui sera multiplie au fur et a mesure
                mat = self.graphe + matrice
                temp = self.graphe + matrice

                #On fait le produit matriciel n-1 fois 
                i = 0
                while i < self.N -1:
                        temp = np.dot(temp,mat)
                        i +=1

                #On obtient au final la matrice (A+In) a la puissance n-1
                #On verifie que tous les coefficients soit different de zeros

                for i in range(self.N):
                        for j in range(i,self.N):
                                if temp[i,j] == 0 :
                                    self.connexe = False
                
                                    return False
                self.connexe = True
                #print temp
                return True
                
                
        def isConnexe2(self):
			matrice =np.eye(self.N)
			mat = self.graphe + matrice
			temp = self.graphe + matrice
			temp2 = self.graphe + matrice
			
			k=0
			while k < self.N -1 :
				for p in xrange(self.N):
					for i in xrange(self.N):
						somme=0
						for j in xrange(self.N):
							somme=somme+temp[i,j]*mat[j,p]
						#on met 1 si on a une position differente de zeros 
						if somme !=0:
							temp[i,p] = 1
						#a la derniere boucle on regarde si on a des zeros si oui on retourne false
						if somme == 0 and k==self.N-2:
							self.connexe = False
							return False
				#actualisation de temp pour le prochain produit matriciel
				temp=np.copy(temp2)
				k+=1
			self.connexe = True
			return True

        def calcShortestPath(self):
            matSP=float("inf")*np.ones((self.N,self.N))
            for i in range(self.N):
                matSP[i,i]=0
            M=np.copy(self.graphe)
            Mt=np.copy(self.graphe)
            for i in range(self.N): #on met la matrice a la puissance
                for j in range(self.N): #on parcoure chaque ligne
                    for k in range(self.N): #et chaque element de la ligne
                        if Mt[j,k]>0 and matSP[j,k]>i+1:
                            matSP[j,k]=i+1
                Mt=np.dot(M,Mt)
            return matSP


        def SCESP(self): #je ne trouve tj pas l'ecart type  :/
            Msp=self.calcShortestPath()
            if self.isConnexe()==False:
                return 2 #on peut majorer la SCE par 2
            mu=np.log(np.log(self.N))
                

                
        def calculCout(self):
                #On recupere les differents couts des differentes methodes
                sce1 = self.sceDegrees()
                sce2 = self.sceCk()
                sce3 = 1

                #On les multiplie entre elles
                cout = sce1*sce2*sce3

                #On met a jour le cout du graphe
                self.cout = cout

                return cout
                



                

                
g=Graphe(10,0.6)
print (g.graphe)
print ("sceDegree:",g.sceDegrees())
print ("sceCk:",g.sceCk())
g.isConnexe()
print ("calculCout" , g.calculCout())

print "la matrice des plus courts chemins est",g.calcShortestPath()


nodes = g.nodesAndEdges()[0]
edges = g.nodesAndEdges()[1]
print ("Edge",edges)
print ("Nodes",nodes)


G=nx.Graph()  
G.add_nodes_from(nodes)
G.add_edges_from(edges)
print G.number_of_nodes()
print G.number_of_edges()
nx.draw(G,node_color="pink") # ROSE bien evidemment ;)
plt.show()



####################################################################################
##                                                                                ## 
##                            CLASS POPULATION                                    ##
##                                                                                ## 
####################################################################################

class Population:
        def __init__(self,taillePop,proba,tailleGraphe):
                self.p=taillePop
                self.proba= proba
                self.population= []
                self.tailleGraphe = tailleGraphe
                for i in range(self.p):
                       self.population.append(Graphe(tailleGraphe,self.proba))

        def mutation(self,matrice):

            for j in range(self.tailleGraphe):
                for k in range(j,self.tailleGraphe):
                        pMutation = random.random()
                        if pMutation > self.proba :
                                if matrice[j,k] == 0:
                                        matrice[j,k] = 1
                                        matrice[k,j] = 1
                                else :
                                        matrice[j,k] = 0
                                        matrice[k,j] = 0
              


        def selection(self):

            bestCost = []
            bestCostSelect = []
            indice = []
            ind = 0
            temp = self.population[0].cout

            for i in range(self.p):
                for j in range(self.p):
                    if i != j : 
                        if self.population[j].cout >= temp :
                            #On verifie que ce cout n est pas deja ete selectionne
                            if j not in indice:
                                temp = self.population[j]
                                ind = j 
                #On remplis notre liste des meilleurs couts.
                bestCost.append(temp)
                indice.append(ind)

            for k in range((self.p)/2):
                bestCostSelect.append(bestCost[k])

            #print bestCostSelect

            return bestCostSelect

        def croisement(self):

            # Duplication de la population avec le meilleur cout

            bestPop = self.selection()
            dupBestPop = []
            for i in range(len(bestPop)):
                dupBestPop.append(bestPop[i])
            #for j in range(len(bestPop)):
            #    dupBestPop.append(bestPop[i])

            #print "dupBestPop Graphe \n",dupBestPop[2].graphe


            # On obtient ainsi une population de graphe avec une duplication des meilleurs
            # Issue de la methode selection qui prend la moitie des meilleurs

            for k in range(4):

                ## MISE EN PLACE DE LA MUTATION  mutation 
                for l in range(len(dupBestPop)):
                    self.mutation(dupBestPop[l].graphe)

                # CROSSING OVER

                ## Mise en place du croisement entre les differents individus 
                # Quel graphe croise avec quel graphe ? 
                l1 = random.randint(0,len(dupBestPop)-1)
                g1 = random.randint(0,len(dupBestPop)-1)
                g2 = random.randint(0,len(dupBestPop)-1)

                while g1 == g2 : 
                    g2 = random.randint(0,len(dupBestPop)-1)
                #print "l1",l1
                #print "Valeurs de g1/g2 \n",g1,g2
                #Puis on choisit aleatoirement les positions qui vont etre croises :
                pos1 = random.randint(0,dupBestPop[0].N-1)
                pos2 = random.randint(0,dupBestPop[0].N-1)
                while pos1 == pos2 : 
                    pos2 = random.randint(0,dupBestPop[0].N-1)

                #print "Valeurs de pos1/pos2 \n",pos1,pos2

                p1 = min(pos1,pos2)
                p2 = max(pos1,pos2)

                # Mise en place du croisement:
                graphe1 = dupBestPop[g1].graphe
                graphe2 = dupBestPop[g2].graphe
                temp1 = np.copy(graphe1)
                temp2 = np.copy(graphe2)

                # print "Les deux graphes a croises sont \n"
                # print "g1" , graphe1
                # print ""
                # print "g2" ,graphe2


                
                for i in range(p1,p2+1):
                    
                    #On modifie le premier graphe
                   
                    graphe1[l1,i] = temp2[l1,i]
                    graphe1[i,l1] = temp2[i,l1]

                    #Puis le deuxieme graphe

                    graphe2[l1,i] = temp1[l1,i]
                    graphe2[i,l1] = temp1[i,l1]

                # Puis on remplace par les nouveaux graphes dans la population que l'on croise.    
                dupBestPop[g1].graphe = graphe1
                dupBestPop[g2].graphe = graphe2


                # print "Les deux graphes a croises sont \n"
                # print "g1" , graphe1
                # print ""
                # print "g2" ,graphe2

            #La nouvelle population contient les anciens meilleurs, puis les nouveaux 
            # Qui sont remanier 
            newPop = []
            newPop.append(bestPop)
            newPop.append(dupBestPop)

        





popu = Population(10,0.6,10)
#print ("Graphe pour le premier individu de la population \n",popu.population[0].graphe)
meilleurcout = popu.selection()
popu.croisement()
# for i in range(len(meilleurcout)):
#     print meilleurcout[i].cout


'''
#pour tester le produit matriciel
G1=Graphe(5,0.6)
g1= G1.graphe


matrice = np.eye(5)
mat = g1 + matrice
temp = g1 + matrice
temp2= g1+matrice
print mat,"\n",temp
			
d=[]
for p in xrange(5):
	for i in xrange(5):
		somme=0
		for j in xrange(5):
			somme=somme+temp[i,j]*mat[j,p]
		if somme !=0:
			temp2[i,p] = somme
		d.append(somme)
temp=np.copy(temp2)
		
print d
print temp

#erreur qui survient de temps en temps

Traceback (most recent call last):
  File "aggpGraphe.py", line 411, in <module>
    popu.croisement()
  File "aggpGraphe.py", line 336, in croisement
    self.mutation(dupBestPop[l].graphe)
AttributeError: 'int' object has no attribute 'graphe'
'''
