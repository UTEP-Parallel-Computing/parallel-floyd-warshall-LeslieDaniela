from mpi4py import MPI
import timeit
import math

# Start Timer
start = timeit.default_timer()

# Read file, assign contents to matrix
with open('fwTest.txt', 'r') as file:
    matrix = [[int(x) for x in line.split()] for line in file]

# File will be used to write results
results_file = open('results.txt', 'w')

# 5x5 Matrix [NOT READING FROM FILE]
# matrix = [[3, 6, 7, 2, 9], [1, 9, 3, 1, 7], [4, 1, 2, 6, 8], [6, 2, 7, 3, 5], [1, 2, 4, 8, 2]]
# print("Matrix to compute: ")
# print(matrix)

# Gets the world communicator
communicator = MPI.COMM_WORLD

# Holds matrix length
matrix_length = len(matrix)

# get the size of the communicator in # processes
rows_per_thread = matrix_length / communicator.Get_size()
threads_per_row = communicator.Get_size() / matrix_length

# Gets rank (process #)
row_start = math.trunc(rows_per_thread * communicator.Get_rank())
row_end = math.trunc(rows_per_thread * (communicator.Get_rank() + 1))

# iterates through file contents
for i in range(matrix_length):
    i_owner = (threads_per_row * i)
    matrix[i] = communicator.bcast(matrix[i], root=i_owner)
    for x in range(row_start, row_end):
        for y in range(matrix_length):
            matrix[x][y] = min(matrix[x][y], matrix[x][i] + matrix[i][y])

if communicator.Get_rank() == 0:
    for i in range(row_end, matrix_length):
        i_owner = (threads_per_row * i)
        # Receive message
        matrix[i] = communicator.recv(source=i_owner, tag=i)

else:
    for i in range(row_start, row_end):
        communicator.send(matrix[i], dest=0, tag=i)

if communicator.Get_rank() == 1:
    print(matrix, file=results_file)

# Ending Timer
stop = timeit.default_timer()
print('Thread: ', communicator.Get_rank())
print('Time: ', stop - start)
