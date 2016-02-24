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
	
	def degrees(self): 
		deg={}
		for i in xrange(self.N+1):
			deg[i]=0
		for i in xrange(self.N):
			deg[sum(self.graphe[i])]+=1
		
		print deg	
		kmin=0
		kmax=self.N
		i=0
		while deg[i]==0:
			kmin+=1
			i+=1
		i=self.N
		while deg[i]==0:
			kmax-=1
			i-=1
		
		th=1/(kmax-kmin)
		sce=0
		for i in xrange(kmin,kmax+1):
			sce+=(th-deg[i])**2
		
		return sce
					
					

g=Graphe(10,0.6)
print g.graphe
g.degrees()
