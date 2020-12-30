import copy
import random
import smmp
import numpy as np
from math import *
from universe1 import *
from protein1 import *
from mergesort import *
from sklearn import preprocessing

phi = np.concatenate((np.random.uniform(-80,-50,10),np.random.uniform(-160,-120,10)))

psi = np.concatenate((np.random.uniform(-50,-20,10),np.random.uniform(110,140,10)))


class geneticAlgorithmProtein():
    '''Genetic Algorithm for python'''
    def __init__(self,max_iteration=100,population_size=100,
                mut_probout=0.18,mut_probin=0.25,crossover_prob=1.,parents_port=0.5,elit_ratio=0.10):
        
        self.dim = len(smmp.var_r.vlvr);
        self.dimcant = smmp.mol_par.nvr

        self.max_iter = max_iteration
        self.pop_size = population_size

        self.mut_prob = mut_probin #0.28
        self.mut_probout = mut_probout #0.25
        self.cross_prob = crossover_prob

        self.parent_port = int(parents_port*self.pop_size)
        trl = self.pop_size - self.parent_port
        if trl%2!=0:
            self.parent_port+=1
        trl = self.pop_size*elit_ratio
        if trl<1 and elit_ratio>0:
            self.num_elit=2
        else: self.num_elit=int(trl)

        self.datazeros=np.zeros(self.dim-self.dimcant)

    def run(self,fitnes):
        self.__fitness = fitnes#-15.0
        
        #cant of angles for aminoacids
        AnglesRes = []
        sumAngle = 0
        for val in smmp.res_i.nvrrs:
            if val == 0:
                break;
            AnglesRes.append([val,sumAngle])
            sumAngle+=val
        
        ######################
        # initial population #
        ######################
        pop = [[np.zeros(self.dim),0]]*self.pop_size

        datazeros=np.zeros(self.dim-self.dimcant)
        for p in range(0,self.pop_size):
            val = copy.deepcopy(np.random.uniform(-180,180,self.dimcant))
            r = random.random()
            val = val + (180-val)*r
            smmp.var_r.vlvr = np.concatenate((val,datazeros))
            pop[p] = [val,myUniverso.energy()]

        pop.sort(key = lambda x: x[1])

        # evaluation chromosoma
        minfit=pop[0][1]
        if self.__fitness >= minfit: return pop[0]

        M_Echrom = copy.deepcopy(pop[:self.num_elit])
        Echrom = copy.deepcopy(M_Echrom[0])

        counter = 0
        while(counter<self.max_iter and Echrom[1] >= self.__fitness):
            Nchrom = self.roulette_wheel_selection(M_Echrom)
            print(counter,Echrom[1])

            # crossover
            offspring1,offspring2 = self.crossoverOne(Echrom,Nchrom,AnglesRes)

            smmp.var_r.vlvr = np.concatenate((offspring1[0],datazeros))     
            offspring1[1]=myUniverso.energy()
            smmp.var_r.vlvr = np.concatenate((offspring2[0],datazeros))
            offspring2[1]=myUniverso.energy()

            #mutation
            if offspring1[1] >= offspring2[1]:
                offspring1 = self.mutation(offspring1,AnglesRes)
                smmp.var_r.vlvr = np.concatenate((offspring1[0],datazeros))
                offspring1[1]=myUniverso.energy()
                if offspring2[1] > Echrom[1]:
                    offspring2 = self.mutation(offspring2,AnglesRes)
                    smmp.var_r.vlvr = np.concatenate((offspring2[0],datazeros))
                    offspring2[1]=myUniverso.energy()
            else:
                offspring2 = self.mutation(offspring2,AnglesRes)
                smmp.var_r.vlvr = np.concatenate((offspring2[0],datazeros))
                offspring2[1]=myUniverso.energy()
                if offspring1[1] > Echrom[1]:
                    offspring1 = self.mutation(offspring1,AnglesRes)
                    smmp.var_r.vlvr = np.concatenate((offspring1[0],datazeros))
                    offspring1[1]=myUniverso.energy()
            
            M_Echrom.append(offspring1)
            M_Echrom.append(offspring2)

            M_Echrom.sort(key = lambda x: x[1])
            M_Echrom = copy.deepcopy(M_Echrom[:self.num_elit])
            Echrom=copy.deepcopy(M_Echrom[0])
            counter+=1

        print("cantidad de iteraciones: ",counter)
        smmp.var_r.vlvr = np.concatenate((Echrom[0],datazeros))
        return Echrom

    def roulette_wheel_selection(self,population):
        population = population[1:self.num_elit]
        fitness = [x[1] for x in population]
        sp = np.sum(fitness)
        vp = []
        for x in population:
            vp.append(x[1]/sp)
        r = random.random()
        vp = preprocessing.normalize([np.array(vp)])
        vp = vp[0]
        idd = np.searchsorted(vp,r,side='right')-1
        #r = random.random()
        #idd = int(idd + (self.num_elit-1-idd)*r)
        Nc = copy.deepcopy(population[idd])

        return Nc

    def crossoverOne(self,x,y,AnglesRes):
        ofs1 = copy.deepcopy(x)
        ofs2 = copy.deepcopy(y)
        # One point
        l = len(AnglesRes)
        ran1=np.random.randint(1,l-1)

        
        for i in range(ran1,l):
            if self.cross_prob > np.random.random() :
                ofs1[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]=y[0][
                    AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]
                ofs2[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]=x[0][
                    AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]
        return ofs1,ofs2

    def crossoverTwo(self,x,y,AnglesRes):
        ofs1 = copy.deepcopy(x)
        ofs2 = copy.deepcopy(y)
        # two point
        l = len(AnglesRes)
        ran1=np.random.randint(0,l)
        ran2=np.random.randint(ran1,l)
        
        for i in range(ran1,ran2):
            if self.cross_prob > np.random.random() :
                ofs1[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]=y[0][
                    AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]
                ofs2[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]=x[0][
                    AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]
        return ofs1,ofs2
        
    def crossoverUniform(self,x,y,AnglesRes):
        ofs1 = copy.deepcopy(x)
        ofs2 = copy.deepcopy(y)
        # uniform
        l = len(AnglesRes)
        
        for i in range(0,l):
            if 0.5 > np.random.random() :
                ofs1[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]=y[0][
                    AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]
                ofs2[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]=x[0][
                    AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]
        return ofs1,ofs2
        
    def crossoverBinary(self,x,y,AnglesRes):
        ofs1 = copy.deepcopy(x)
        ofs2 = copy.deepcopy(y)
        # uniform
        l = len(AnglesRes)
        
        eta=2
        for i in range(0,l):
            ran = np.random.random()
            if 0.5 > ran :
            	beta = 2.*ran
            else:
            	beta = 1./(2.*(1.-ran))
            	
            eta **= 1. / (eta + 1.)
            x1 = y[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]
            x2 = x[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]
            
            ofs1[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]] = 0.5 * (((
            								1 + beta) * x1) + ((1 - beta) * x2))
            ofs2[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]] = 0.5 * (((
            								1 - beta) * x2) + ((1 + beta) * x1))

        return ofs1,ofs2


    def mutation(self,x,AnglesRes):
        ofs = copy.deepcopy(x)                                                    
        l = len(AnglesRes)
        for i in range(0,l):
            if self.mut_probout > np.random.random():
                for j in range(AnglesRes[i][0]):
                    if self.mut_prob > np.random.random():
                        index = AnglesRes[i][1]+j
                        #r = random.random()
                        replace = np.random.uniform(-180,180)
                        #replace = -180 + (180+180)*r
                        ofs[0][index]=replace
                    
        return ofs


myUniverso = Universe(T=300,st=0)
protA = Protein("EXAMPLES/prueba.seq",'')
myUniverso.add(protA)

GAP = geneticAlgorithmProtein(50000,200,
                            mut_probout= 0.6,#0.18,#0.2
                            mut_probin= 0.06, #0.25, #0.15
                            elit_ratio=0.5)
Echrom = GAP.run(-15)

print(myUniverso.energy(),myUniverso.rgyr(),myUniverso.helix(),protA.hbond())

smmp.outpdb(0,'final.pdb')

