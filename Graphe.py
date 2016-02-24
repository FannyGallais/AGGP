import numpy as np
import random

class Graphe:
	def __init__(self,tailleGraphe,p):
		self.N=tailleGraphe
		self.cout=0
		self.connexe=False
		self.graphe=np.zeros((self.N,self.N))
		
		#while !self.graphe.isConnexe():
		for i in xrange(self.N):
			for j in xrange(i,self.N):
				if random.random()<p and i != j:
					self.graphe[i,j]=1
					self.graphe[j,i]=1
	
	def isConnexe(self):
		return True
		
					
					

g=Graphe(4,0.6)
print g.graphe
