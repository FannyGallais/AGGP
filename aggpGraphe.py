import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

import Population

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

        
        def coutMin_Max_Moy(self):
                couts=[]
                for i in range(len(self.pop.population)):
                        #couts.append(self.pop.population[i].calculCout())
                        couts.append(self.pop.population[i].cout)
                        
                cout_Moy=sum(couts)*1.0/len(couts)
                return min(couts),max(couts),cout_Moy
                
                
                
        def generation(self):
                compt=0 #compteur

                #print ("seuil",self.seuil)
                coutMin=[]
                coutMoy=[]
                while compt < self.seuil:
                        #SELECTION
                        print compt
                        #print "selection"
                        popSelect=self.pop.selection()
                        couts=self.coutMin_Max_Moy()
                        coutMin.append(couts[0])
                        coutMoy.append(couts[2])
                        #partie de la pop qui va muter
                        popSelect2=popSelect[1]
                        
                        #CROISEMENT
                        #print "croisement"
                        self.pop.croisement(popSelect2,self.nbCrois)
                        
                        #MUTATION
                        #print "mutation"
                        for i in range(len(popSelect2)):
                                self.pop.mutation(popSelect2[i].graphe)
                                
                        #MAJ DE LA POP
                        #print "maj"
                        
                        self.pop.population[0]=popSelect[0]
                        self.pop.population[1:]=popSelect2[:]
                        compt+=1
                        print ("nombre de tour",compt)
                        

                self.pop.saveInFile(self.coutMin_Max_Moy(),compt)
                
                return coutMin,coutMoy

        
                        

def drawSCE(g):
	#sce DEGRES
	plt.figure("Degres")
	sceDeg=g.sceDegrees()
	plt.plot(range(len(sceDeg[1])),sceDeg[1],marker='o',color='cyan')
	plt.plot(range(len(sceDeg[2])),sceDeg[2],marker='v',color='purple')
	plt.title("Distribution theorique de la somme des carres des ecarts / Distibution observee")

	#sce Ck
	plt.figure("Ck")
	sceCk=g.sceCk()
	plt.plot(range(len(sceCk[1])),sceCk[1],marker='o',color='cyan')
	plt.plot(range(len(sceCk[2])),sceCk[2],marker='v',color='purple')
	plt.title("Distribution theorique de la somme des carres des ecarts / Distibution observee")

	"""
	#sce SP
	plt.figure("Shortest Path")
	sceSP=g.SCESP()[1]
	l1=sorted(sceSP.keys())
	l2=sceSP.values()
	mu=np.log(np.log(g.N))
	plt.plot(range(100),[mu]*100,marker='o',color='cyan')
	plt.plot(l1,l2,"v",color='purple')
	plt.title("Distribution theorique de la somme des carres des ecarts / Distibution observee")
	"""
	plt.show()



#g=Graphe(50,0.7)
#drawSCE(g)

popu = Population.Population(50,0.1,0.8,100)

<<<<<<< HEAD
                
"""
              
g=Graphe(10,0.6)


print (g.graphe)
print ("sceDegree:",g.sceDegrees())
print ("sceCk:",g.sceCk())
g.isConnexe()
print ("calculCout" , g.calculCout())

print "la matrice des plus courts chemins est",g.calcShortestPath()
print g.SCESP()


it = range(1,12)
plt.plot(it,g.sceDegrees()[1],marker='o',color='cyan')
plt.plot(it,g.sceDegrees()[2],marker='v',color='purple')
plt.title("Distribution theorique de la somme des carres des ecarts / Distibution observee")
#plt.legend([p1,p2],["Theorique","Observee"])
plt.show()
=======
>>>>>>> refs/remotes/origin/master



simul=Simulation(popu,100,20)
result=simul.generation()

x=range(len(result[0]))
plt.figure("figure 1")
plt.plot(x,result[0],color="red")
plt.plot(x,result[1],color="blue")
plt.show()

<<<<<<< HEAD
"""

####################################################################################
##                                                                                ## 
##                            CLASS POPULATION                                    ##
##                                                                                ## 
####################################################################################

class Population:
        def __init__(self,taillePop,proba,tailleGraphe,c=0.4):
                self.p=taillePop
                self.proba= proba
                self.population= []
                self.Wr=[]
                for i in xrange(self.p):
					self.Wr.append(self.p*((c-1)/(c**self.p-1))*c**(self.p-i))
				
                self.tailleGraphe = tailleGraphe
                for i in range(self.p):
					#print i
					self.population.append(Graphe(tailleGraphe,self.proba))
					self.Wr[i]=self.Wr[i]/sum(self.Wr)
                       
                
				
				   


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
              


        def selection(self,c=0.4):
			Cost = []
			bestCost=[]
			dico={}
			bestCostSelect=[]
			for i in xrange(self.p):
				self.population[i].calculCout()

          

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
			
			
		
			rang=np.random.multinomial(self.p-1,self.Wr)
			
			bestCostSelect.append(bestCost[0])
			for i in xrange(len(rang)):
				j=rang[i]
				while j!=0:
					bestCostSelect.append(bestCost[i])
					j-=1
			
				
			return bestCostSelect
			
			
			

        def croisement(self,dupBestPop,nbCrois):
			for k in xrange(nbCrois):

				# CROSSING OVER

				## Mise en place du croisement entre les differents individus 
				# Quel graphe croise avec quel graphe ? 
				l1 = random.randint(0,dupBestPop[0].N-1)
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
		
		
	def generation(self):
		compt=0 #compteur
		while compt < self.seuil:
			#SELECTION

			popSelect=self.pop.selection()

			#CROISEMENT
			self.pop.croisement(popSelect,self.nbCrois)
			
			#MUTATION
			for i in xrange(len(popSelect)):
				self.pop.mutation(popSelect[i].graphe)
			
			#MAJ DE LA POP
			for k in xrange(self.pop.p):
				self.pop.population[k]=popSelect[k]
				
			compt+=1
			#print "nombre de tour",compt

	
			




popu = Population(50,0.9,50)





simul=Simulation(popu,50,5)
simul.generation()

g = simul.pop.population[0]
#print g

nodes = g.nodesAndEdges()[0]
edges = g.nodesAndEdges()[1]
#print ("Edge",edges)
#print ("Nodes",nodes)
=======
g = simul.pop.population[0]
drawSCE(g)

nodes = g.nodesAndEdges()[0]
edges = g.nodesAndEdges()[1]
>>>>>>> refs/remotes/origin/master

plt.figure("figure 2")

G=nx.Graph()  
G.add_nodes_from(nodes)
G.add_edges_from(edges)
<<<<<<< HEAD
#print G.number_of_nodes()
#print G.number_of_edges()
nx.draw(G,node_color="pink") # ROSE bien evidemment ;)
plt.show()

#print g.sceDegrees()[1]



it = range(1,51)

#print len(it)
distributionTheo = list(g.sceDegrees()[2])
distributionTheo = distributionTheo[0:50]
print len(distributionTheo)
print len(g.sceDegrees()[1])


plt.plot(it,g.sceDegrees()[1],marker='o',color='cyan')
plt.plot(it,distributionTheo,marker='v',color='purple')
plt.title("Distribution theorique de la somme des carres des ecarts / Distibution observee")
#plt.legend([p1,p2],["Theorique","Observee"])
plt.show()





=======
print (G.number_of_nodes())
print (G.number_of_edges())
nx.draw(G,node_color="pink") # ROSE bien evidemment ;)
plt.show()
>>>>>>> refs/remotes/origin/master

"""

# for i in range(len(meilleurcout)):
#     print meilleurcout[i].cout



#pour tester le produit matriciel
G1=Graphe(5,0.6)
g1= G1.graphe


matrice = np.eye(5)
mat = g1 + matrice
temp = g1 + matrice
temp2= g1+matrice
print mat,"\n",temp
                        
d=[]
for p in range(5):
        for i in range(5):
                somme=0
                for j in range(5):
                        somme=somme+temp[i,j]*mat[j,p]
                if somme !=0:
                        temp2[i,p] = somme
                d.append(somme)
temp=np.copy(temp2)
                
print d
print temp
'''

#erreur qui survient de temps en temps
'''
Traceback (most recent call last):
  File "aggpGraphe.py", line 411, in <module>
    popu.croisement()
  File "aggpGraphe.py", line 336, in croisement
    self.mutation(dupBestPop[l].graphe)
AttributeError: 'int' object has no attribute 'graphe'
<<<<<<< HEAD
'''
=======


"""
>>>>>>> refs/remotes/origin/master
