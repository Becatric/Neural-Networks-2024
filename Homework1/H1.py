from trace import Trace


def parseEquation(line):
    parts = line.split()
    if parts[0] == '-':
        valOfX = -int(parts[0].replace('x', '').replace('-', '') or '1')
    else:
        valOfX = int(parts[0].replace('x', '') or '1')
    sign_y = 1 if parts[1] == '+' else -1
    valOfY = sign_y * int(parts[2].replace('y', '') or '1')
    sign_z = 1 if parts[3] == '+' else -1
    valOfZ = sign_z * int(parts[4].replace('z', '') or '1')
    resultVal = int(parts[6])
    return (valOfX, valOfY, valOfZ, resultVal)


def readEquations(filename):
    A = []
    B = []
    with open(filename, 'r') as file:
        for line in file:
            valX, valY, valZ, resultValue = parseEquation(line)
            a = []
            a.append(valX)
            a.append(valY)
            a.append(valZ)
            A.append(a)
            B.append(resultValue)
    return (A, B)


def determinant(A):
    det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][
        2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    return det


def determinant2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def trace(A):
    trace = A[0][0] + A[1][1] + A[2][2]
    return trace


def vectorNorm(B):
    vectNorm = (B[0] ** 2 + B[1] ** 2 + B[2] ** 2) ** (1 / 2)
    return vectNorm


def transpose(A):
    AT = [[0] * len(A) for _ in range(len(A[0]))]
    x = len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            AT[j][i] = A[i][j]
    return AT


def matrixMultiply(A, B):
    if len(A[0]) != len(B):
        raise ValueError("Number of columns in A must match the number of elements in B")
    C = [0] * len(A)
    for i in range(len(A)):
        row_sum = 0
        for j in range(len(B)):
            row_sum += A[i][j] * B[j]
        C[i] = row_sum
    return C


def replaceColumn(A, B, col):
    AT = [[0] * len(A) for _ in range(len(A[0]))]

    for i in range(len(A)):
        for j in range(len(A)):
            if (j == col):
                AT[i][j] = B[i]
            else:
                AT[i][j] = A[i][j]
    return AT


def cramerRule(A, B):
    x, y, z = 0, 0, 0
    Ax = replaceColumn(A, B, 0)
    Ay = replaceColumn(A, B, 1)
    Az = replaceColumn(A, B, 2)
    x = determinant(Ax) / determinant(A)
    y = determinant(Ay) / determinant(A)
    z = determinant(Az) / determinant(A)
    return x, y, z


def get_minor(A, i, j):
    minor = []
    for row in range(len(A)):
        if row != i:
            minor_row = []
            for col in range(len(A[row])):
                if col != j:
                    minor_row.append(A[row][col])
            minor.append(minor_row)
    return minor


def cofactor_matrix(matrix):
    cofactors = []
    for i in range(len(matrix)):
        cofactor_row = []
        for j in range(len(matrix)):
            minor = get_minor(matrix, i, j)
            cofactor = ((-1) ** (i + j)) * determinant2(minor)
            cofactor_row.append(cofactor)
        cofactors.append(cofactor_row)

    return cofactors


def InverseMatrix(matrix):
    inverse = [[0] * len(A) for _ in range(len(A[0]))]
    cofactor = transpose(cofactor_matrix(matrix))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            inverse[i][j] = (1 / determinant(matrix)) * cofactor[i][j]
    return inverse


# ###############################################


A, B = readEquations('E:/RN/Homework1/InputFile.txt')
print(" A :")
for row in A:
    print(row)

print("\n B :")
print(B)
#
# print("\n", determinant(A))
# print("\n", trace(A))
# print("\n", vectorNorm(B))
# print("\n", transpose(A))
# print("\n", matrixMultiply(A,B))
# C = replaceColumn(A,B,0)
# print(" C :", C)
# print(cramerRule(A, B))
# get_minor(A,1,1)
inverse = InverseMatrix(A)
print(inverse)



# Bonus
# Cofactorul e dezvoltare dupa ficare element
# pe cand Determinantul lui A e dezvoltare dupa elementele de pe prima linie
# deci foloseste la baza aceeasi formula