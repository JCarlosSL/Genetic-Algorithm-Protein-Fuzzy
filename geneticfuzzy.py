import copy
import random
import smmp
import numpy as np
from math import *
from universe1 import *
from protein1 import *

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

    def run(self,fitness):
        self.__fitness = fitness#-15.0
        
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
            # crossover
            print(counter,Echrom[1])
            offs = self.crossoverfuzzy(M_Echrom,counter,AnglesRes)

            for i in range(len(offs)):
                #print(offs[i][1],end=' ')
                smmp.var_r.vlvr = np.concatenate((offs[i][0],datazeros))     
                offs[i][1]=myUniverso.energy()
                if offs[i][1]>Echrom[1]:
                    offs[i]=self.mutation(offs[i],AnglesRes)
                    smmp.var_r.vlvr = np.concatenate((offs[i][0],datazeros))     
                    offs[i][1]=myUniverso.energy()
                M_Echrom.append(offs[i])
            #print()
            M_Echrom.sort(key = lambda x: x[1])
            M_Echrom = copy.deepcopy(M_Echrom[:self.num_elit])
            Echrom=copy.deepcopy(M_Echrom[0])
            counter+=1

        print("cantidad de iteraciones: ",counter)
        smmp.var_r.vlvr = np.concatenate((Echrom[0],datazeros))
        return Echrom
        
    def crossoverOne(self,x,y,AnglesRes):
        ofs1 = copy.deepcopy(x)
        ofs2 = copy.deepcopy(y)
        # One point
        l = len(AnglesRes)
        ran1=np.random.randint(0,l)
        
        for i in range(0,ran1):
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

    def crossoverfuzzy(self,population,counter,AnglesRes):
        ca,p = self.calculate_ca_and_p(population)
        ca=1.-ca
        p=1.-p
        #print(ca,p)
        male, female = self.separate_by_gender(population,counter)
        population_size = len(population)
        total_offspring = int(round(population_size * (p * 1.0))) // 2
        offspring = []
        for x in range(total_offspring):
            a = self.female_tournament_selection(female, 5)
            b = self.male_selection(male, a)
            if ca < 0.375:#35
                #offspring = self.two_pc(a, b, offspring)
                offspring = self.i_c(a, b, offspring)
            elif ca < 0.626: #70
                offspring = self.k_pc(a, b, AnglesRes, offspring)
            else:
                #offspring = self.i_c(a, b, offspring)
                offspring = self.two_pc(a, b, offspring)
        return offspring

    def two_pc(self,a, b, AnglesRes, output_array=None):
        if output_array is None:
            output_array = []
        #length = len(a[0])
        length = len(AnglesRes)
        first_point = np.random.randint(length)
        second_point = np.random.randint(length)
        if first_point > second_point:
            tmp = first_point
            first_point = second_point
            second_point = tmp

        x = copy.deepcopy(a)
        y = copy.deepcopy(b)
        for i in range(first_point,second_point):
                x[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]=a[0][
                        AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]
                y[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]=b[0][
                        AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]
        output_array.append(x)
        output_array.append(y)
        return output_array


    def k_pc(self,a, b,AnglesRes, output_array=None):
        if output_array is None:
            output_array = []

        length = len(AnglesRes)

        x = copy.deepcopy(a)
        y = copy.deepcopy(b)
        iterator = 0
        for i in range(length):
            if random.random()<0.5:
                x[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]=a[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]
                y[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]=b[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]

        output_array.append(x)
        output_array.append(y)
        return output_array

    def i_c(self,a, b,AnglesRes, output_array=None):
        if output_array is None:
            output_array = []
        #length = len(a[0])
        length = len(AnglesRes)
        first_point = np.random.randint(length)
        second_point = np.random.randint(length)
        if first_point > second_point:
            tmp = first_point
            first_point = second_point
            second_point = tmp

        x = copy.deepcopy(a)
        y = copy.deepcopy(b)
        for i in range(length):
            if (i < first_point) or (i > second_point):
                x[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]=a[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]
                y[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]=b[0][AnglesRes[i][1]:AnglesRes[i][0]+AnglesRes[i][1]]
            else:
                x[0][AnglesRes[length-i-1][1]:AnglesRes[length-i-1][0]+AnglesRes[length-i-1][1]]=b[0][AnglesRes[length-i-1][1]:AnglesRes[length-i-1][0]+AnglesRes[length-i-1][1]]
                y[0][AnglesRes[length-i-1][1]:AnglesRes[length-i-1][0]+AnglesRes[length-i-1][1]]=a[0][AnglesRes[length-i-1][1]:AnglesRes[length-i-1][0]+AnglesRes[length-i-1][1]]
        output_array.append(x)
        output_array.append(y)
        return output_array

    def male_selection(self,male, female_chro, size=-1):
        if size == -1:
            size = len(male) // 2 
        #male = [ma[0] for ma in male1]
        male_temp = random.sample(male, size)
        male_hamming_distance = [self.hammingdistance(x, female_chro) for x in male_temp]
        male_fitness_value = [x[1] for x in male_temp]
        male_active_genes = [self.active_genes(x) for x in male_temp]
        male_index = [x for x in range(len(male_temp))]
        male_chro_index = max(zip(male_hamming_distance, male_fitness_value, male_active_genes, male_index))[3]
        male_chro = male_temp[male_chro_index]
        return male_chro
    def active_genes(self,a):
        return np.sum(a[0])

    def female_tournament_selection(self,female, tournament_round):
        tournament_round -= 1
        female_chro = female[np.random.randint(0, len(female))]
        fitness_value_female_chro = female_chro[1]
        for i in range(tournament_round):
            chromosome = female[np.random.randint(0, len(female))]
            fitness_value_temp = chromosome[1]
            if fitness_value_temp > fitness_value_female_chro:
                female_chro = chromosome
                fitness_value_female_chro = fitness_value_temp

        return female_chro

    def separate_by_gender(self,population, generation):
        gender = 0
        generation = generation % 2
        female = []
        male = []
        for x in population:
            if (gender % 2) != generation:
                female.append(x)
            else:
                male.append(x)
            gender = gender + 1
        return male, female

    def calculate_ca_and_p(self,population):
        t1,t2,t3 = self.calculate_t(population)
        #print(t1,t2,t3)
        t1_low, t1_medium, t1_high = self.calculate_t1_membership(t1)
        t2_low, t2_high = self.calculate_t2_membership(t2)
        t3_low, t3_medium, t3_high = self.calculate_t1_membership(t3) ##ne

        ca_low_array = []
        ca_medium_array = []
        ca_high_array = []

        p_low_array = []
        p_medium_array = []
        p_high_array = []

        # rule 1
        ca_high_array.append(max(t1_low, t2_low, t3_low))
        p_high_array.append(min(t1_low, t2_low, t3_low))

        # rule 2
        ca_high_array.append(max(t1_low, t2_low, t3_medium))
        p_high_array.append(min(t1_low, t2_low, t3_medium))

        # rule 3
        ca_medium_array.append(max(t1_low, t2_low, t3_high))
        p_medium_array.append(min(t1_low, t2_low, t3_high))

        # rule 4
        ca_high_array.append(max(t1_low, t2_high, t3_low))
        p_medium_array.append(min(t1_low, t2_high, t3_low))

        # rule 5
        ca_medium_array.append(max(t1_low, t2_high, t3_medium))
        p_medium_array.append(min(t1_low, t2_high, t3_medium))

        # rule 6
        ca_medium_array.append(max(t1_low, t2_high, t3_high))
        p_low_array.append(min(t1_low, t2_high, t3_high))

        # rule 7
        ca_high_array.append(max(t1_medium, t2_low, t3_low))
        p_high_array.append(min(t1_medium, t2_low, t3_low))

        # rule 8
        ca_medium_array.append(max(t1_medium, t2_low, t3_medium))
        p_medium_array.append(min(t1_medium, t2_low, t3_medium))

        # rule 9
        ca_medium_array.append(max(t1_medium, t2_low, t3_high))
        p_medium_array.append(min(t1_medium, t2_low, t3_high))

        # rule 10
        ca_medium_array.append(max(t1_medium, t2_high, t3_low))
        p_medium_array.append(min(t1_medium, t2_high, t3_low))

        # rule 11
        ca_medium_array.append(max(t1_medium, t2_high, t3_medium))
        p_medium_array.append(min(t1_medium, t2_high, t3_medium))

        # rule 12
        ca_medium_array.append(max(t1_medium, t2_high, t3_high))
        p_low_array.append(min(t1_medium, t2_high, t3_high))

        # rule 13
        ca_high_array.append(max(t1_high, t2_low, t3_low))
        p_high_array.append(min(t1_high, t2_low, t3_low))

        # rule 14
        ca_medium_array.append(max(t1_high, t2_low, t3_medium))
        p_medium_array.append(min(t1_high, t2_low, t3_medium))

        # rule 15
        ca_low_array.append(max(t1_high, t2_low, t3_high))
        p_medium_array.append(min(t1_high, t2_low, t3_high))

        # rule 16
        ca_low_array.append(max(t1_high, t2_high, t3_low))
        p_medium_array.append(min(t1_high, t2_high, t3_low))

        # rule 17
        ca_low_array.append(max(t1_high, t2_high, t3_medium))
        p_low_array.append(min(t1_high, t2_high, t3_medium))

        # rule 18
        ca_low_array.append(max(t1_high, t2_high, t3_high))
        p_low_array.append(min(t1_high, t2_high, t3_high))    
        u_ca_x = sum([self.calculate_ca_low_x(y) * y for y in ca_low_array])
        u_ca_x += sum([self.calculate_ca_medium_x(y) * y for y in ca_medium_array])
        u_ca_x += sum([self.calculate_ca_high_x(y) * y for y in ca_high_array])
        u_ca = sum(ca_low_array)
        u_ca += sum(ca_medium_array)
        u_ca += sum(ca_high_array)
        ca = u_ca_x / (u_ca * 1.0)

        u_p_x = sum([self.calculate_p_low_x(y) * y for y in p_low_array])
        u_p_x += sum([self.calculate_p_medium_x(y) * y for y in p_medium_array])
        u_p_x += sum([self.calculate_p_high_x(y) * y for y in p_high_array])
        u_p = sum(p_low_array)
        u_p += sum(p_medium_array)
        u_p += sum(p_high_array)
        p = u_p_x / (u_p * 1.0)
        return ca, p

    def calculate_ca_low_x(self,y):
        if y == 1.0:
            return 0.25
        if y == 0.0:
            return 1.0
        x = (0.25 * y) + 0.25
        return x

    def calculate_ca_medium_x(self,y):
        if y == 0.0:
            return 0.0
        return 0.5

    def calculate_ca_high_x(self,y):
        if y == 1.0:
            return 0.75
        if y == 0.0:
            return 0.5
        return (0.25 * y) + 0.5

    def calculate_p_low_x(self,y):
        return self.calculate_ca_low_x(y)

    def calculate_p_medium_x(self,y):
        return self.calculate_ca_medium_x(y)

    def calculate_p_high_x(self,y):
        return self.calculate_ca_high_x(y)

    def calculate_t1_membership(self,t1):
        #low = 1.0
        low = 0.0
        medium = 0.0
        #high = 0.0
        high = 1.0

        if t1 <= 0.25:
            low = 1.0
        elif t1 <= 0.5:
            low = (-4 * t1) + 2

        if t1 <= 0.25:
            medium = 0.0
        elif t1 <= 0.5:
            medium = (4 * t1) - 1
        elif t1 <= 0.75:
            medium = (-4 * (t1 - 0.25)) + 2

        if t1 <= 0.5:
            high = 0.0
        elif t1 <= 0.75:
            high = (4 * (t1 - 0.25)) - 1

        return low, medium, high

    def calculate_t2_membership(self,t2):
        #low = 1.0
        low = 0.0
        #high = 0.0
        high = 1.0

        if t2 <= 0.25:
            low = 1.0
        elif t2 <= 0.5:
            low = (-4 * t2) + 2

        if t2 <= 0.25:
            high = 0.0
        elif t2 <= 0.5:
            high = (4 * t2) - 1

        return low, high

    def calculate_t(self,population):
        size = len(population)
        allfitness = [x[1] for x in population]
        maxfitness = max(allfitness)*1.0
        minfitness = min(allfitness)*1.0
        averagefitness = np.mean(allfitness)*1.0
        chromaxfitness = population[allfitness.index(maxfitness)]
        chrominfitness = population[allfitness.index(minfitness)]
        uniquefitness = len(set(allfitness))
        t1 = uniquefitness / (size*1.0)
        t2 = (maxfitness - averagefitness)/(maxfitness*1.0)
        t3 = self.hammingdistance(chromaxfitness,chrominfitness)/(len(chromaxfitness)*1.0)
        return t1,t2,t3

    def hammingdistance(self,a, b):
        distance = 0
        #print(a)
        for i in range(len(a)):
            if a[0][i] != b[0][i]:
                distance += 1
        return distance

    def mutation(self,x,AnglesRes):
        ofs = copy.deepcopy(x)                                                    
        l = len(AnglesRes)
        for i in range(0,l):
            if self.mut_probout > np.random.random():
                for j in range(AnglesRes[i][0]):
                    if self.mut_prob > np.random.random():
                        index = AnglesRes[i][1]+j
                        #val = ofs[0][index]
                        #data = val+(180-val)*random.random()
                        replace = np.random.uniform(-180.0,180.0)#val)
                        ofs[0][index]=replace
                    
        return ofs


myUniverso = Universe(T=300,st=0)
protA = Protein("EXAMPLES/1bdd.seq",'')
myUniverso.add(protA)

GAP = geneticAlgorithmProtein(50000,150,
                            mut_probout= 0.22,#0.18,#0.2
                            mut_probin= 0.18, #0.25, #0.15
                            elit_ratio=0.14)
Echrom = GAP.run(-15)

print(myUniverso.energy(),myUniverso.rgyr(),myUniverso.helix(),protA.hbond())

smmp.outpdb(0,'final.pdb')

