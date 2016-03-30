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
	# plt.figure("Degres")
	# sceDeg=g.sceDegrees()
	# plt.plot(range(len(sceDeg[1])),sceDeg[1],marker='o',color='cyan')
	# plt.plot(range(len(sceDeg[2])),sceDeg[2],marker='v',color='purple')
	# plt.title("Distribution theorique de la somme des carres des ecarts / Distibution observee")
    
	#sce Ck

	plt.figure("Ck")
	sceCk=g.sceCk()
	plt.plot(range(len(sceCk[1])),sceCk[1],marker='o',color='cyan')
	plt.plot(range(len(sceCk[2])),sceCk[2],marker='v',color='purple')
	plt.title("Distribution theorique de la somme des carres des ecarts / Distibution observee")

	
	# #sce SP
	# plt.figure("Shortest Path")
	# sceSP=g.SCESP()[1]
	# l1=sorted(sceSP.keys())
	# l2=sceSP.values()
	# mu=np.log(np.log(g.N))
	# plt.plot(range(100),[mu]*100,marker='o',color='cyan')
	# plt.plot(l1,l2,"v",color='purple')
	# plt.title("Distribution theorique de la somme des carres des ecarts / Distibution observee")
	# """
	plt.show()



#g=Graphe(50,0.7)
#drawSCE(g)

popu = Population.Population(50,0.1,0.8,30)




simul=Simulation(popu,150,20)
result=simul.generation()

# x=range(len(result[0]))
# plt.figure("figure 1")
# plt.plot(x,result[0],color="red")
# plt.plot(x,result[1],color="blue")
# plt.show()

g = simul.pop.population[0]
drawSCE(g)

nodes = g.nodesAndEdges()[0]
edges = g.nodesAndEdges()[1]

plt.figure("figure 2")

G=nx.Graph()  
G.add_nodes_from(nodes)
G.add_edges_from(edges)
#print (G.number_of_nodes())
#print (G.number_of_edges())
nx.draw(G,node_color="pink") # ROSE bien evidemment ;)
plt.show()

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

#erreur qui survient de temps en temps

Traceback (most recent call last):
  File "aggpGraphe.py", line 411, in <module>
    popu.croisement()
  File "aggpGraphe.py", line 336, in croisement
    self.mutation(dupBestPop[l].graphe)
AttributeError: 'int' object has no attribute 'graphe'


"""
