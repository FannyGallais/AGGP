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
                self.proba=p
                while self.connexe==False:
					for i in xrange(self.N):
						for j in xrange(i,self.N):
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
            for i in xrange(self.N):
                    nodes.append(i)
                    for j in xrange(i,self.N):
                        if self.graphe[i,j]==1:
                            edges.append((i,j))
            return (nodes,edges)


        def degrees(self):
                deg={}
                for i in xrange(self.N):
                        deg[i]=0 #on met toutes les valeurs du dico a 0
                for i in xrange(self.N):
                        deg[sum(self.graphe[i])]+=1

                return deg
        
        def sceDegrees(self,gamma=2.5):
                deg = self.degrees()
                print ("deg",deg)

                th=[0]*self.N
                for i in xrange(1,self.N):
                    
                        th[i-1] = i**(-gamma)
                #print (th)
                c=sum(th)
                

                for i in xrange(1,self.N):
					th[i-1]=th[i-1]/c

                print (th)
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
                for i in xrange(kmin,kmax):
                        sce+=(th[i]-(deg[i]/self.N))**2

                
                print "deg",deg.values()*(1/self.N)

                return (sce,th)


        
        def sceCk(self):
                
                Ci=[0]*self.N #stocke les coefficients de clustering pour chaque noeud
                ni=[0]*self.N

                #pour chaque noeuds on garde en memoire ses voisins, grace a un dictionnaire : dicoN
                dicoN = {}
                for i in xrange(self.N):
                        dicoN[i]=[]
                for i in xrange(self.N):
                        for j in xrange(self.N):
                                if self.graphe[i,j]==1:
                                        dicoN[i].append(j)
                print("dicoN:",dicoN)

                #on stocke dans ni le nombres de liens entre les voisins d'un noeuds
                for i in xrange(self.N):
                        n=0
                        if len(dicoN[i])!=0 and len(dicoN[i])!=1: #si le noeud n'a aucun ou un seul voisin ni vaudra 0 et Ci aussi
                                for j in xrange(len(dicoN[i])-1): 
                                        for k in xrange(j+1,len(dicoN[i])):
                                                #print(j,k,dicoN[i],"\n")
                                                if self.graphe[dicoN[i][j],dicoN[i][k]]==1:
                                                        n+=1
                        ni[i]=n

                print ("ni:",ni)
                
                for i in xrange(self.N):
                        if sum(self.graphe[i])!=0 and sum(self.graphe[i])!=1:
                                Ci[i]=2*ni[i]/(sum(self.graphe[i])*(sum(self.graphe[i])-1))
                #print (Ci)

                Ck={}
                for i in xrange(self.N+1):
                        Ck[i]=0 
                for i in xrange(self.N+1):
                        somme = 0
                        k=0
                        for j in xrange(self.N):
                                if sum(self.graphe[j])==i:
                                        somme=somme+Ci[j]
                                        k+=1
                        if k!=0:
                                Ck[i]= somme/k

                #print (Ck)
                sce=0
                for i in xrange(len(Ck)-1):
                        sce+=(1/(i+1)-Ck[i+1])**2
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

                for i in xrange(self.N):
                        for j in xrange(i,self.N):
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
            for i in xrange(self.N):
                matSP[i,i]=0
            M=np.copy(self.graphe)
            Mt=np.copy(self.graphe)
            for i in xrange(self.N): #on met la matrice a la puissance
                for j in xrange(self.N): #on parcoure chaque ligne
                    for k in xrange(self.N): #et chaque element de la ligne
                        if Mt[j,k]>0 and matSP[j,k]>i+1:
                            matSP[j,k]=i+1
                Mt=np.dot(M,Mt)
            return matSP


        def SCESP(self): #je ne trouve tj pas l'ecart type  :/
            Msp=self.calcShortestPath()
            if self.isConnexe()==False:
                return  #on peut majorer la SCE par 2
            mu=np.log(np.log(self.N))
            d={} #on cree le dico avec toutes les cles (lg du chemin) et le nb d'occurence de ces longueurs dans le dico
            for i in range(self.N):
				for j in range(i+1,self.N):
					lgP=Msp[i,j]
					if Msp[i,j] in d.keys():
						d[lgP]+=1
					else:
						d[lgP]=1
            cout=0
            for lg in d.keys():
                cout+=abs(float(lg)-mu)*float(d[lg])/float(self.N)
            return cout
                

                
        def calculCout(self,a=1,b=1,c=1):

                #On recupere les differents couts des differentes methodes
                sce1 = self.sceDegrees()[0]
                sce2 = self.sceCk()
                sce3 = self.SCESP()


                #On les ajoute entre elles
                cout = (a*sce1)+(b*sce2)+(c*sce3)

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
print g.SCESP()

it = range(1,11)
plt.plot(it,g.sceDegrees()[1])
plt.title("Distribution theorique de la somme des carres des ecarts / Distibution observee")
plt.show()
"""

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

"""

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

			Cost = []
			bestCost=[]
			bestCostSelect = []
			dico={}

			for i in xrange(self.p):
				Cost.append(self.population[i].cout)
				if self.population[i].cout in dico:
					dico[self.population[i].cout].append(i)
				else:
					dico[self.population[i].cout]=[i]
					
			Cost=sorted(Cost)
			
			for k in xrange(len(Cost)):
				indGraphe=dico[Cost[k]][0]
				bestCost.append(self.population[indGraphe])
				dico[Cost[k]].remove(indGraphe)
		

			for j in xrange((self.p)/2):
				bestCostSelect.append(bestCost[j])

			return bestCostSelect

        def croisement(self,dupBestPop,nbCrois):
			"""
            # Duplication de la population avec le meilleur cout

            bestPop = self.selection()
            dupBestPop = []
            for i in range(len(bestPop)):
                dupBestPop.append(bestPop[i])
            #for j in xrange(len(bestPop)):
            #    dupBestPop.append(bestPop[i])

            #print "dupBestPop Graphe \n",dupBestPop[2].graphe


            # On obtient ainsi une population de graphe avec une duplication des meilleurs
            # Issue de la methode selection qui prend la moitie des meilleurs
"""
			for k in xrange(nbCrois):
				"""
				## MISE EN PLACE DE LA MUTATION  mutation 
				for l in xrange(len(dupBestPop)):
					self.mutation(dupBestPop[l].graphe)
				"""
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


				
				for i in xrange(p1,p2+1):
					
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
			#newPop = []
			#newPop.append(bestPop)
			#newPop.append(dupBestPop)
			return dupBestPop
        def saveInFile(self,nameFile="info.txt"):
			f=open(nameFile,"w")
			f.write("taillePop: %f\n"%self.p)
			f.write("tailleGraphe: %f\n"%self.tailleGraphe)
			f.write("Proba: %f\n"%self.proba)
			f.close()


####################################################################################
##                                                                                ## 
##                            CLASS SIMULATION                                    ##
##                                                                                ## 
####################################################################################


class Simulation:
	def __init__(self,pop,seuil,nbCrois):
		self.pop=pop
		self.seuil=seuil
		self.nbCrois=nbCrois

	
	def coutMin_Max(self):
		couts=[]
		for i in xrange(len(self.pop.population)):
			couts.append(self.pop.population[i].cout)
		return min(couts),max(couts)
		
	# def arret(self):
	# 	stop=False
	# 	#print "difference coutMax-coutMin",self.coutMin_Max()[1]-self.coutMin_Max()[0]
	# 	if self.coutMin_Max()[1]-self.coutMin_Max()[0]<self.seuil:
	# 		stop=True
	# 	return stop
		
	def duplicate(self,l):
		l2=[]
		for i in xrange(len(l)):
			G=Graphe(l[i].N,l[i].proba)
			G.graphe=np.copy(l[i].graphe)
			l2.append(G)
		return l2
		
	def generation(self):
		compt=0 #compteur
		while compt < self.seuil:
			#SELECTION

			popSelect=self.pop.selection()
			
			temp=self.duplicate(popSelect)
			#temp=np.copy(popSelect)

			#MUTATION
			for i in xrange(len(popSelect)):
				self.pop.mutation(popSelect[i].graphe)
			
			#CROISEMENT
			self.pop.croisement(popSelect,self.nbCrois)
			
			#MAJ DE LA POP
			for k in xrange(self.pop.p/2):
				self.pop.population[k]=temp[k]
				self.pop.population[k+self.pop.p/2]=popSelect[k]
			compt+=1
			print "nombre de tour",compt

	
			



popu = Population(20,0.6,10)






simul=Simulation(popu,10,5)
simul.generation()

g = simul.pop.population[0]
print g

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



# for i in xrange(len(meilleurcout)):
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
