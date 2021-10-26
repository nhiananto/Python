# =============================================================================
# numpy reference
# =============================================================================

import numpy as np

#creation
a = np.array([1,2,3,4])
b = np.array((1,2,3,4))
c = np.array([[1,2,3],[4,5,6]], dtype = float)

#arange is simlar to range, but returns an array
a = np.arange(15).reshape(3,5)

#properties
a.ndim
a.shape
a.shape[0]
a.dtype
len(a)
a.size #size of all elements

np.append(b,4)
np.append(4,b)
np.append(a, 4) #append will flatten to 1d array if does not provide axis
np.append(a, [[4,4,4,4,4]], axis = 0)
np.append(a, [[4],[4],[4]], axis = 1)

#fill the array/change the values
c.fill(0)

# ==========================
#placeholder matrix
# ==========================
np.zeros((3,5))
np.ones((3,4,5))
np.full((3,5), 99) #with fill value

#identity
np.identity(5)
np.eye(5)
np.eye(5,3) #can also make non-square "identity"

#similar matrix
a = np.arange(15).reshape(3,5)
np.ones_like(a)
np.zeros_like(a)
np.ones_like(a, shape = (5,3)) #reshaped

# ==========================
#create diagonals
# ==========================
np.diag(np.arange(1,6))

#get diagonals
np.diagonal(a, offset = 0)
np.diagonal(a, offset = 1)
np.diagonal(a, offset = -1)
#can also provide axis for the diagonals

# ==========================
#sequences
# ==========================
np.arange(5) #same as np.arange(0,5)
np.arange(8,15,3)
np.arange(start = 5, stop = 12, step = 2, dtype = float)

#create evenly spaced arrays (similar to length.out)
np.linspace(start = 0, stop = 1, num = 10)

# ==========================
# repeats
# ==========================
np.repeat([1,2,3], 3)
np.repeat([1,2,3], [1,2,3]) #repeat different number of times for different numbers
np.repeat(np.array([1,2,3]).reshape(-1,1), 3, axis = 1) #can repeat along different axis

#repeat each = F
b = np.arange(1,4)
b.repeat(3).reshape(-1,3, order = 'F').ravel()

# tile
# "tiles"/stacking copies along a given axis
t = np.random.randn(2,2)
np.tile(t, reps = 2) #tile to the "right"
np.tile(t, reps = (1,2)) #same as above
np.tile(t, reps = (2,1))
np.tile(t, reps = (2,2))

# ==========================
#matrix
# ==========================
c = np.array([[1,2,3],[4,5,6]], dtype = float)
np.asmatrix(c)
#matrix are the same as 2D arrays with:
#always 2D (even when flattened)
#over-rides multipication to be matrix multiplication
#over-rides power to be matrix power
#matrices are a subclass of the ndarray

# =============================================================================
# conversions
# =============================================================================
a = np.arange(10)

#convert to list
a.tolist()

#dtype casting
a.astype(float)
a.astype(complex)
a.astype(str)

# =============================================================================
# reshaping
# =============================================================================
a = np.arange(15).reshape(3,5)
a.reshape(-1,3) #-1 means whatever needed
a.reshape(5,3)
np.reshape(a, (5,3))

#by row (default)
np.arange(1,16).reshape(3,5, order = 'C')
#by column
np.arange(1,16).reshape(3,5, order = 'F')

#convert to 1d array
a = np.arange(15).reshape(3,5)
np.ravel(a)
#same as
np.reshape(a, -1)
#convert by col
np.ravel(a, order = 'F')
np.reshape(a, -1, order = 'F')

#flatten to 1d
a.flatten()
a.flatten(order = 'F')

#transpose
a.T
a.transpose()
#can give axes for n-D array
d = np.arange(1,9).reshape(2,2,2)
d.transpose(0,2,1) 
d.transpose(1,2,0)
d.transpose(2,0,1)

#resize modifies it inplace
c = c.resize(3,2) 


# =============================================================================
# vectorization
# =============================================================================
#by elements operation
a**2
b = a * 2
a - b
a < 10

A = np.array([[1,1],[0,1]])
B = np.array([[2,0],[3,4]])
A, B
#element wise multiplication
A * B
#matrix multiplication
A @ B
#or dot
np.dot(A,B)
A.dot(b)

# =============================================================================
# stacking (cbind/rbind)
# =============================================================================
a = np.ones((2,3))
b = np.zeros((2,3))

#column stacking (cbind)
np.hstack((a,b))
np.c_[a,b] #convenient wrapper/helper

#rbind
np.vstack((a,b))
np.r_[a,b]

#for 1D arrays
c = np.ones(3)
d = np.zeros(3)

#different from hstack for 1D arrays
print(np.column_stack((c,d)))
print(np.hstack((c,d)))

#same in 1D and 2d arrays
np.vstack((c,d))

#row_stack is the same as vstack (row_stack is alias for vstack)

#combining/stacking across any axis
np.concatenate((c,d))
np.concatenate((a,b), axis = 0) #same as vstack
np.concatenate((a,b), axis = 1) #same as hstack


#np.dstack (3-D /3rd axis)
f = np.ones((3,2,3))
g = np.zeros((3,2,3))
f,g
np.vstack((f,g))
np.hstack((f,g))
np.dstack((f,g))
np.concatenate((f,g), axis = 2) #same as dstack

#splitting
c = np.arange(1,10)
np.split(c, 3)
np.split(c,4) #will not allow non-equal sizes
#can also provide axis for splitting

#will allow near equal size splitting
np.array_split(c,4)

# =============================================================================
# slicing, indexing, selection
# =============================================================================
a = np.arange(10)
a[0:3] #item 0 1 and 2
a[2:5] #item 2 - 4
a[:7] #from start to 6

a[[4,6,7]] #multiple items
a[np.arange(1,3).tolist() + [6,7]] #in R c(1:2, 6,7)

#from 0 to 8 every second item
a[:8:2]
a[np.arange(0, 8, 2)]

a[-1] #last item
a[-2] #second last
a[4:-1] #4 to 1 before last

a[::] #all
a[::2] #step by 2
a[::-1] #reverse step

#setting
a[0:3] = 100
a

#multi dimensional
b = np.arange(1,16).reshape(5,3)
b
b[2,2] #3rd row, 3rd column
b[:,2] #3rd column
b[::2,2] #every 2 steop for row, 3rd column

b[1:3, :]

x = np.array([[1, 2], [3, 4], [5, 6]])
x
x[[0, 1, 2], [0, 1, 0]] #from row 0 1 and 2 choose items 0 1 and 0 respectively

x = np.array([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]])
x[[0,0,3,3], [0,2,0,2]] #select corner elements returns 1D array
x[ [[0,0], [3,3]], [[0,2], [0,2]] ] #returns 2d array

#newaxis
x = np.arange(5)
x[:,np.newaxis] #make into 2d
x[:,np.newaxis, np.newaxis] #3d


#where (returns index)
a = np.arange(10)
np.where(a < 5)
a[np.where(a < 5)]
a[a<5]
np.where([1,2,4] == a)

#any
np.any(a == [1,100])

#take (same as fancy indexing)
np.take(b, [[1,3],[2,5]])

#put, fills in the value (same as fancy indexing)
np.put(b, [1,3], 100)

#roll (rolls the array)
np.roll(a,2)

#diff calculates difference between subsequent elements
np.diff(a)

#iterating multi dimensional array iterates by row
for row in b:
    print(row)

for row in b:
    for col in row:
        print(col)

#can flatten
for element in b.flat:
    print(element)

#accessing each b element
for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        print(b[i,j])

# =============================================================================
# unary functions and apply
# =============================================================================
a = np.arange(1,21)
b = a*3
np.sum(a)
np.mean(a)
np.sqrt(a)
np.exp(a)
np.log(a)
np.cumsum(a)
np.cumprod(a)
np.min(a)
np.max(a)
np.floor(a)
np.ceil(a)
np.std(a)
np.var(a)
np.cov(a,b)
np.corrcoef(a,b)
np.round(a,3)

#compares 2 arrays element by element
np.minimum(a,b)
np.maximum(a,b)

#some functions are provided as ndarray methods
#all have axis arguments (except round)
a.sum()
a.mean()
a.cumsum()
a.cumprod()
a.min()
a.max()
a.std()
a.var()
a.round(3)

#along axis operations
b = np.arange(1,16).reshape(3,5)
b.sum(1) # b.sum(axis = 1)
b.sum(0)
b.sum(axis = 1, keepdims = True) #keepdims argument

b.cumsum(1) #along row
b.min(1)
b.std(1)

#max/min along a given axis
np.amax(b, 0)
np.amax(b, 1)
np.amin(b, 1)

#apply function
np.apply_along_axis(np.sum, 1, b)
np.apply_along_axis(np.var, 0, b)

#see also apply_over_axis
#applys a function over multiple axes
c = np.arange(24).reshape(2,3,4)
# Sum over axes 0 and 2. The result has same number of dimensions as the original array:
np.apply_over_axes(np.sum, c, [0,2])
#equivalent to tuple axis arguments with keepdims = True
c.sum(axis = (0,2), keepdims = True)

#vectorizes a function (NOT FOR PERFOMANCE, ESSENTIALLY A FOR LOOP)
def myfunc(a,b):
    "Return a-b if a>b, otherwise return a+b"
    if a > b:
        return a - b
    else:
        return a + b
myfunc(a,b) #does not work
vmyfunc = np.vectorize(myfunc)
vmyfunc(a,b)


#cut into deciles
a = np.arange(1,21)
np.percentile(a, np.arange(1,11)*10, interpolation  = 'higher')
# =============================================================================
# logic functions
# =============================================================================
np.isnan(a)
np.isinf(a)
np.isfinite(a)

x = np.arange(10)
y = np.arange(10) * 2

x == y    
np.equal(x,y)

x <= y
np.less_equal(x,y)

x != y
np.not_equal(x,y)
