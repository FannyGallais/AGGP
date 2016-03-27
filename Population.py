import numpy as np
import random
#import networkx as nx
import matplotlib.pyplot as plt
import Graphe
####################################################################################
##                                                                                ## 
##                            CLASS POPULATION                                    ##
##                                                                                ## 
####################################################################################

class Population:
        def __init__(self,taillePop,probam,probai,tailleGraphe,c=0.6):
                self.p=taillePop
                self.proba= probam #la proba de mutation
                self.probaI=probai #la proba d'initialisation
                self.population= []
                self.Wr=[]
                for i in range(self.p):
                        self.Wr.append(self.p*((c-1)/(c**self.p-1))*c**(self.p-i))
                                
                self.tailleGraphe = tailleGraphe
                for i in range(self.p):
                        print (i)
                        self.population.append(Graphe.Graphe(tailleGraphe,probai,"G"+str(i))) #la proba d'initialisation
                        #print (self.population[i])
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
                        #population a muter
                        bestCostMuter=[]
                        
                        for i in range(self.p):
                                self.population[i].calculCout()
                        

                        for i in range(self.p):
                                Cost.append(self.population[i].cout)
                                if self.population[i].cout in dico:
                                        dico[self.population[i].cout].append(i)
                                else:
                                        dico[self.population[i].cout]=[i]
                                        
                        Cost=sorted(Cost,reverse=True)
                        #print (test)
                    
                        #print ("minCout",Cost[-1])
                        
                        for k in range(len(Cost)):
                                indGraphe=dico[Cost[k]][0]
                                bestCost.append(self.population[indGraphe])
                                dico[Cost[k]].remove(indGraphe)
                        
                        
                        rang=np.random.multinomial(self.p-1,self.Wr)
                        #print (rang)

                        #le meilleur de la pop qui lui ne sera pas mute
                        best = bestCost[-1]
                        for i in range(len(rang)):
                                j=rang[i]
                                while j!=0:
                                        bestCostMuter.append(Graphe.Graphe(self.tailleGraphe,self.probaI,bestCost[i].nom,bestCost[i]))
                                        j-=1
                        #print ("meilleur graphe :",best,best.cout)
                        
                        return best,bestCostMuter
                        
                        
                        

        def croisement(self,dupBestPop,nbCrois):
                        for k in range(nbCrois):

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


                        return dupBestPop
                        
        def saveInFile(self,cout,compt,nameFile="info.txt"):
                        f=open(nameFile,"w")
                        f.write("taillePop: %f\n"%self.p)
                        f.write("tailleGraphe: %f\n"%self.tailleGraphe)
                        f.write("Proba mutation: %f\n"%self.proba)
                        f.write("Proba initialisation:%f\n"%self.probaI )
                        f.write("cout min: %f\n"%cout[0])
                        f.write("cout max: %f\n"%cout[1])
                        f.write("compteur: %d\n"%compt)
                        f.close()



                        """
                        print (bestCost[6].graphe[2,6])
                        bestCost[7]=bestCost[6]
                        if bestCost[6].graphe[2,6]==1:
                            bestCost[6].graphe[2,6]=0
                        else :
                            bestCost[6].graphe[2,6]=1
                        
                        print (bestCost[7].calculCout(),bestCost[6].calculCout())
                        """
