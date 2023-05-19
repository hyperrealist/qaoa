from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

import numpy as np

NumberOfQubits = 12

Q = np.zeros((NumberOfQubits, NumberOfQubits), dtype=int)
for i in range(NumberOfQubits):
    for j in range(i + 1, NumberOfQubits):
        if np.random.randint(0, 2) == 1:
            Q[i][j] = Q[j][i] = - 1
    Q[i][i] = - np.sum(Q[i])

NumberOfComponents = NumberOfQubits - np.linalg.matrix_rank(Q)

BetaGrid = 3
GammaGrid = 3
p = 3

grid = []
for i in range(BetaGrid):
    for j in range(GammaGrid):
        grid.append([np.pi * i / BetaGrid, 2 * np.pi * j / GammaGrid])

AnglePaths = [[]]
for i in range(p):
    NewAnglePaths = []
    for path in AnglePaths:
        for pair in grid:
            NewAnglePaths.append(path + [pair])
    AnglePaths = NewAnglePaths

results = []
step = 1
total_steps = len(AnglePaths)

for angles in AnglePaths:
    
    QRegX = QuantumRegister(len(Q))
    CLRegX = ClassicalRegister(len(Q))
    
    QC = QuantumCircuit(QRegX, CLRegX)
    
    QC.h(QRegX)
    
    for pair in angles:
        
        # Cost bang
        for i, q in enumerate(QRegX):
            if Q[i][i] != 0:
                QC.p(Q[i][i] * pair[1], q)
            for j, r in enumerate(QRegX[:i]):
                if Q[i][j] != 0:
                    QC.cp(2 * Q[i][j] * pair[1], q, r)
    
        # Mixer bang
        QC.rx(pair[0],QRegX)
        
    QC.measure(QRegX, CLRegX)
    
    simulator = AerSimulator()
    compiled_QC = transpile(QC, simulator)    
    counts = simulator.run(compiled_QC, shots=1).result().get_counts(compiled_QC)
    
    for s in counts:
        conf = np.array([int(x) for x in s[::-1]])
    
    results.append([conf, np.matmul(conf, np.matmul(Q, np.transpose(conf))), angles])
    print("\t\tprogress\t= " + str(100 * step / total_steps)[:5] + "%")
    step += 1

results.sort(key = lambda x: x[1])
for result in results[-5:]:
    print("\n\t\tconfigutaion\t=",result[0].view())
    print("\t\tcut\t\t=",result[1])
    print("\t\tangles\t\t=",result[2])

MaxCut = 0
for x in range(1 << (NumberOfQubits - 1)):
    conf = np.array([(x >> i)&1 for i in range(NumberOfQubits)])
    MaxCut = max(MaxCut, np.matmul(conf, np.matmul(Q, np.transpose(conf))))

print("\n\tthe graph (number of components =",str(NumberOfComponents) + ", number of edges = " + str(np.trace(Q) // 2) + ", maxcut = " + str(MaxCut) + ")\n")
for row in Q:
    print("\t",row.view())
print()