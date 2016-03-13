import numpy as np
import random

class Graphe:
        def __init__(self,tailleGraphe,p):
                self.N=tailleGraphe
                self.cout=0
                self.connexe= True
                self.graphe=np.zeros((self.N,self.N))
		#while !self.graphe.isConnexe():
                for i in range(self.N):
                        for j in range(i,self.N):
                                if random.random()<p and i != j:
                                        self.graphe[i,j]=1
                                        self.graphe[j,i]=1
        def isConnexe(self):
                return True

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
                print temp
                

                
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
                



                

                
               

                
g=Graphe(5,0.6)
print (g.graphe)
print ("sceDegree:",g.sceDegrees())
print ("sceCk:",g.sceCk())
g.isConnexe()
print ("calculCout" , g.calculCout())

class Population:
        def __init__(self,taillePop,proba,tailleGraphe):
                self.p=taillePop
                self.proba= proba
                self.population= []
                self.tailleGraphe = tailleGraphe
                for i in range(self.p):
                       self.population.append(Graphe(tailleGraphe,self.proba))


popu = Population(5,0.6,3)
print popu.population[1].graphe




