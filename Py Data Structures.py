# =============================================================================
# Python Data Structures
# =============================================================================

# =============================================================================
# dict
# =============================================================================
a = {}
type(a)

#add new items
a["a"] = 1
a.update({"b" : 2})
a.update({"c" : 3, "d": 4}) #update can also update existing key value pair


#can add
a["a"] += 1


#get values from key
a.get("a")
a.get("z") #return none if key does not exist

a.update({"e": a.get("e", 0) + 1}) #if e doesn't exist yet use 0 as default from a.get()

#get list of keys
a.keys()


#checking if key exists
if "a" in a.keys(): print("exist")
if "a" in a: print("exist")
if a.get("z", -1) == -1: print("does not exist")


a.items() #return (key, value) pairs
[k for k, v in a.items() if v < 3]

len(a)

#dict can contain other data types as values
a.update({"a" : (3, 2)})


#dict is sorted as of python 3.7
#remove key and return its value
a.pop("a", default = None) #return default value if given if key error, if not raise KeyError


#return last key, value pair
a.popitem()

#dictionary to count items in a list
a = ['apple','red','apple','red','red','pear']
d = {item : a.count(item) for item in a}
print(d)

#dict comprehension
ct = {k : 0 for k in "abc"}
print(ct)

#remove all items
a.clear()

# =============================================================================
# set (unordered)
# =============================================================================
a = set(['a', 'a', 'b', 'c'])

print(a) #only contains unique values (unordered)
#useful to get unique values

#remove set items
a.remove('a')
#a.pop() #removes last element (random since unordered)


#add set items
a.add('a')

b = set(['b', 'd'])


## set operations
a.union(b) # a + b not defined

a - b #in a but not in b
a.difference(b)

a | b #in a or b

a & b #in a and b
a.intersection(b)

a ^ b #in a or b but not both (XOR)

a.isdisjoint(b)
b.issubset(a) #or a.issuperset(b) #superset -> opposite of subset (i.e. if a contains all elements of b)


#check if an item is in a set
'a' in a





# =============================================================================
# deque (double ended queue) can be used as either stack/queue
# =============================================================================
from collections import deque
a = deque(['a', 'b', 'c'])


a.popleft()
a.pop()


a.appendleft('b')
a.append('b')


a.extendleft(['a', 'b'])
a.extend(['a', 'b'])


# =============================================================================
# array/list
# =============================================================================
a = []
a.append('a')

a.extend(['a', 'b', 'c', 'd']) #"unpacks" the iterable before appending it
a.append(['a', 'b', 'c', 'd']) #compare

a.pop() #pop last element added (can be used as stack)
a.pop(0) #pop first element instead of last (can used as queue)

a.insert(0, 'a')

a.remove('a') #remove specific value

#sort
a.extend(['z', 'k'])
a.sort() #sort in-place

#reverse
a[::-1] #not-in place
a.reverse() #in-place


#map (returns iterator object)
list(map(lambda x: x[0], a)) #get first element inside list a

# =============================================================================
# heap (min heap)
# for max heap push negative (-) values
# =============================================================================
import heapq

a = [1, 10, 4, 0, -1]

heapq.heapify(a) #transform a populated list to a heap
print(a) #sorted heap

heapq.heappush(a, -10)
print(a)


heapq.heappop(a) #return lowest priority
print(a)

heapq.heappush(a, 10) # can push same priority


heapq.heappushpop(a, -10) #push first then pop (heap size stays fixed)
heapq.heapreplace(a, -10) #pop first then push (heap size stays fixed)


#can also push tuples
a = []

heapq.heappush(a, (5, 10))

heapq.heappush(a, (5, 11)) #if the first value of the tuple is the same, will compare second value and so on
print(a) #the second value must also be comparable (int, float, or "str")


#can also push (int, object) as long as int is unique
a = []

heapq.heappush(a, (5, {"a" : 1}))

heapq.heappush(a, (6, {"a" : 1})) #if first tuple is unique, second element will not be checked

heapq.heappush(a, (5, {"b" : 2})) #if first value tuple is same, will return error since second value cannot be compared



#heap sort
#push all values, then pop one at a time
def heapsort(iterable):
        h = []
        for value in iterable:
            heapq.heappush(h, value)
        return [heapq.heappop(h) for i in range(len(h))]


# =============================================================================
# priority queue (partial wrapper for heapify, thread-safe)
# =============================================================================
from queue import PriorityQueue

a = PriorityQueue()

a.queue #return list of what is in queue
a.empty()

#push
a.put(10) 
a.put(-1) 
a.put(5) 
a.put(100) 
a.queue


a.get() #pop


# =============================================================================
# tuples (immutable ordered lists)
# =============================================================================
a = (1, 2, 3)

a[1] = 4 #immutable








