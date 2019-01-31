import csv
import copy, math
from operator import itemgetter
import random
from random import uniform as rand
from random import randint
from deap import algorithms, base, creator, tools
import numpy as np
import matplotlib.pyplot as plt

WO = []
WO2 = []
n=[]


with open('Workorder Prioritisation and Scheduling - Sheet3.csv', newline='\n') as csvfile:
    sampleSet = csv.reader(csvfile, delimiter = '\n', quotechar='|')      
    for row in sampleSet:
        l = row[0]
        WO.append(l.split(','))
         
for i in range(1, len(WO)):
    n.clear()
    for ii in range(0, len(WO[i])):
        element = WO[i][ii]        
        if element == '':
           m = 0
        else: 
           m = float(element)
        n.append(m)
    n.append(0)
    n.append(0)
    n.append(0)
    n.append(0)
       
    WO2.append(copy.deepcopy(n))
    
WO3 = copy.deepcopy(WO2) #WO2 hold the initial order of the items for later reference

def NORMCDF(x, u, l): #Outputs the cumulative probability for x where x is part of a normal distribution with mean u and standard deviation l
    y = 0.5*(1+math.erf((x-u)/(l*(2**0.5))))
    return y

def NORM(x, u, l):
    y = (1/math.sqrt(2*math.pi*l**2))*math.exp(-((x-u)**2)/(2*l**2))
    return y 

def equivalentCostInOrder(WO2):
    totalCost = 0 
    
    for i in range(0, len(WO2)):

        itemCost = WO2[i][1]/WO2[i][4]
        totalCost = totalCost + itemCost
        WO2[i][7] = itemCost
        
    return totalCost

def riskInOrder(WO2): #Evaluates the permutation using the complete list
    totalRisk = 0
    startDate = 0
    
    for i in range(0, len(WO2)):
        duration = WO2[i][6]
        WO2[i][7] = startDate
        itemRisk = NORMCDF(startDate, (WO2[i][4]+WO2[i][2]), WO2[i][3]) *WO2[i][1] +WO2[i][5]
        totalRisk = totalRisk + itemRisk
        WO2[i][8] = itemRisk
        startDate = startDate + duration

    return totalRisk

def riskInOrder2(WOindices): #Evalutes the permutation using the indices only. Added to allow compatibility with DEAP
    a = []
    for i in range (0, len(WOindices)):
        a.append(WO3[WOindices[i]])
        
    b = riskInOrder(a)        
    return (b,)
    

def swapLowHigh(WO2):
    for ii in range(1, len(WO2)):
        for i in range(1, len(WO2)-ii):
            previousCost = riskInOrder(WO2)
            WO3 = copy.deepcopy(WO2)
            if WO2[i][7] < WO2[i+ii][7]:
               a = WO2[i]
               WO2[i] = WO2[i+ii]
               WO2[i+ii] = a
            if riskInOrder(WO2) > previousCost:
               WO2 = WO3
            
    return(WO2)

def swapHighLow(WO2):
    for ii in range(1, len(WO2)):
        for i in range(1, len(WO2)-ii):
            
            previousRisk = riskInOrder(WO2)
            WO3 = copy.deepcopy(WO2)
            
            if WO2[i][7] >= WO2[i+ii][7]:
               a = WO2[i]
               WO2[i] = WO2[i+ii]
               WO2[i+ii] = a
            if riskInOrder(WO2) > previousRisk:
               WO2 = WO3
    return(WO2)

print("Current Evaluation :", riskInOrder(WO2))

def randomPermutation(WOindices): #Generates a randomly permutated list the same size as the list is it fed. 
    a = []
    b = []
    for i in range(0, len(WOindices)):
        a.append(i)
    for i in range(0, len(WOindices)):
        c = randint(0, len(a)-1)
        b.append(a[c])
        del a[c]   
    return b

def simpleOptimiser(WO2):
    #Quick algorithm so find a slightly optimised solution
    for x in range(0,1):
    
        WO2 = sorted(WO2, key=itemgetter(5)) #Order shortest duration first
        #print(equivalentCostInOrder(WO2), " ", riskInOrder(WO2))

        a = riskInOrder(WO2)
        WO2 = swapHighLow(WO2)
        WO2 = swapLowHigh(WO2)
        b = riskInOrder(WO2)

        while (b<a): #Iterates until no further improvement is found
            a = riskInOrder(WO2)
            WO2 = swapHighLow(WO2)
            WO2 = swapLowHigh(WO2)
            b = riskInOrder(WO2)
            print("Current Evaluation :", riskInOrder(WO2))
    return WO2

def addFailureDate(WO2):
    totalTime = 0
    for items in WO2:                  
        totalTime = totalTime + items[6]

    for items in WO2:        
        for i in range(0, int(totalTime+1)):           
            if i > totalTime:
                items[9] = totalTime
            else: 
                pFailure = NORMCDF(i, (items[4]+items[2]), items[3])
                r = random.random()                        
                if r > pFailure:
                    items[9] = i
            
    return WO2

def costNoBreakIn(WO2):   #The final bill, if something breaks and you dont change schedule 
    a = []    
    for i in range(0, 100):
        totalCost = 0
        WO2 = addFailureDate(WO2)
        for items in WO2:
            if items[7]-items[6] > items[9]:
                items[10] = int((items[1]/items[6])*(items[7]-items[9]-items[6])+items[5]) #Cost/day*number of days from failure to fixed
                totalCost = totalCost + items[10]
            else:
                items[10] = items[5]
        a.append(totalCost)
            
    return (np.min(a), np.mean(a), np.max(a))

def costWithBreakIn(WO2):   #The final bill, if something breaks and you change the schedule to fix it sooner
    a = []    
    for i in range(0, 100):
        totalCost = 0
        WO2 = addFailureDate(WO2)
        for items in WO2:
            if items[7]-items[6] > items[9]:
                items[10] = int((items[1]/items[6])*(items[7]-items[9]-items[6])+items[5]) #Cost/day*number of days from failure to fixed
                totalCost = totalCost + items[10]
            else:
                items[10] = items[5]
        a.append(totalCost)
            
    return (np.min(a), np.mean(a), np.max(a))
        
      
        
def geneticOptimiser(WO2):#Genetic algorithms part
        
    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox.register("indices", randomPermutation, WOindices)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb = 0.05)
    toolbox.register("evaluate", riskInOrder2)
    toolbox.register("select", tools.selTournament, tournsize = 3)

    statistics = {"Avg": np.mean,
                  "Std": np.std,
                  "Max": np.max,
                  "Min": np.min}

    fit_stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    for Name, func in statistics.items():    
        fit_stats.register(Name, func)

    pop = toolbox.population(n = 65)
    result, log = algorithms.eaSimple(pop, toolbox, cxpb = 0.8, mutpb = 0.1, ngen = 200, verbose = False, stats = fit_stats)
    best_individual = tools.selBest(result, k=1)[0]

    print("Current Genetic Evaluation :", riskInOrder2(best_individual))   
    
    WO2 = []

    for i in range (0, len(best_individual)):
            WO2.append(WO3[ (best_individual[i]-1)])

    plt.figure(figsize=(11, 4))
    plt.plot(log.select("Avg"), label = "Mean", c  = "r")
    plt.plot(np.add(log.select("Avg"),log.select("Std")),c  = "r", linestyle = "--")
    plt.plot(np.subtract(log.select("Avg"),log.select("Std")), c  = "r", linestyle = "--")
    plt.plot(log.select("Min"), label = "Min", c  = "b")
    plt.plot(log.select("Max"), label = "Max", c  = "g")
    plt.legend()
    plt.ylabel('Total cost')
    plt.xlabel('Iterations');
    plt.show()
            
    return WO2

#Main#

WOindices = randomPermutation(WO2)
WO3 = addFailureDate(WO3)
print("Cost No Break-in :", costNoBreakIn(WO2), "\n")
WO2 = simpleOptimiser(WO2)

for i in range(0, len(WO2)): #Take the slightly optimised list and store the order in a list called WOindices 
    WOindices[i] = WO2[i][0]

#WO2 = geneticOptimiser(WO2)
#WO2 = simpleOptimiser(WO2)
print("Cost No Break-in :", costNoBreakIn(WO2), "\n")

#

a = []
for items in WO2:
    a.append(items[0])

for items in WO2:
    print(items)




    

        



    
