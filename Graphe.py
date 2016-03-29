import numpy as np
import random
#import networkx as nx
import matplotlib.pyplot as plt
class Graphe:
        def __init__(self,tailleGraphe,p,name,g=0):
                self.N=tailleGraphe
                self.cout= random.randint(0,self.N) # En attendant d'avoir le vrai cout
                self.connexe= False
                self.nom=name
                self.p=p
                if g==0:
                        self.graphe=np.zeros((self.N,self.N))
                        while self.connexe==False:
							for i in range(self.N):
									for j in range(i,self.N):
											if random.random()<p and i != j:
													self.graphe[i,j]=1
													self.graphe[j,i]=1
							self.connexe=self.isConnexe()
                                        #pour verifier que les deux methodes isConnexe donnent les memes resultats
                        
                else:
                        self.graphe=np.copy(g.graphe)
        

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
                
                for i in range(self.N+1):
                        deg[i]=0 #on met toutes les valeurs du dico a 0
                for i in range(self.N):
                        deg[sum(self.graphe[i])]+=1
                

                return deg



        
        def sceDegrees(self,gamma=2.5):
                deg = self.degrees()
                #print ("deg",deg)

                th=[0]*self.N
                for i in range(1,self.N):
                    
                        th[i-1] = i**(-gamma)
                #print (th)
                c=sum(th)
                

                for i in range(1,self.N):
                                        th[i-1]=th[i-1]/c

                #print (th)
                kmin=0
                kmax=self.N-1
                # i=0
                # while deg[i]==0:
                #         kmin+=1
                #         i+=1
                # i=self.N-1
                # while deg[i]==0:
                #         kmax-=1
                #         i-=1
                        
                sce=0
                for i in range(kmin,kmax):
                        sce+=(th[i]-(deg[i]/self.N))**2
                
                degObs = np.asarray(list(deg.values()))*(1/float(self.N))
                #print "deg",degObs

                return (sce,th,degObs)


        
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
                #print("dicoN:",dicoN)

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

                #print ("ni:",ni)
                
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
                #print "Dictionnaire",Ck

                #print (Ck)
                sce=0
                th = []
                th.append(0)
               

                for i in range(len(Ck)-1):
                    th.append(1/float((i+1)))

                    sce+=(th[i+1]-Ck[i+1])**2

                result = (sce,th,Ck.values())
                return result

                

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
                return True
                
                
        def isConnexe2(self):
                        matrice =np.eye(self.N)
                        mat = self.graphe + matrice
                        temp = self.graphe + matrice
                        temp2 = self.graphe + matrice
                        
                        k=0
                        while k < self.N -1 :
                                for p in range(self.N):
                                        for i in range(self.N):
                                                somme=0
                                                for j in range(self.N):
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
                return (self.N*(self.N-1))/2 #on peut majorer la SCE par 2 PROBLEME QUE FAIT ON QUAND C PAS CONNEXE
            mu=np.log(np.log(self.N))
            d={} #on cree le dico avec toutes les cles (lg du chemin) et le nb d'occurence de ces longueurs dans le dico
            for i in range(self.N):
                for j in range(i+1,self.N):
                        lgP=Msp[i,j]
                        if Msp[i,j] in d.keys():
                                d[lgP]+=1
                        else:
                                d[lgP]=1
            cost=0
            for lg in d.keys():
                cost+=abs(float(lg)-mu)*float(d[lg])/float(self.N)
            results=(cost,d)
            return cost
            
                

                
        def calculCout(self,a=1,b=1,c=1):

                #On recupere les differents couts des differentes methodes
                sce1 = self.sceDegrees()[0]
                sce2 = self.sceCk()[0]
                #print ("erreur SCESP()[0]:",self.SCESP())
                sce3 = self.SCESP()
                #print "sceDeg",sce1,"sceCk",sce2,"sceSP",sce3
                #print sce1,sce2,sce3

                #On les ajoute entre elles
                cout = (a*sce1)+(b*sce2)+(c*sce3)

                #On met a jour le cout du graphe
                self.cout = cout

                return cout
            
        def __str__(self):
            return self.nom                                

        def __repr__(self):
            return self.nom
                


"""
g=Graphe(10,0.6,1)
print (g.graphe)
g1=Graphe(10,0.6,1,g)
print (g1.graphe)
g1.graphe[0,0]=12
print (g1.graphe)
print(g.graphe)


                

print (g.graphe)
print ("sceDegree:",g.sceDegrees())
print ("sceCk:",g.sceCk())
g.isConnexe()
print ("calculCout" , g.calculCout())

print "la matrice des plus courts chemins est",g.calcShortestPath()
print g.SCESP()

it = range(1,11)
plt.plot(it,g.sceDegrees()[1],marker='o',color='cyan')
plt.plot(it,g.sceDegrees()[2],marker='v',color='purple')
plt.title("Distribution theorique de la somme des carres des ecarts / Distibution observee")
#plt.legend([p1,p2],["Theorique","Observee"])
plt.show()
"""
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
