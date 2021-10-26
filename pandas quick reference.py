"""
Pandas Quick Reference
pandas v1.1.3
"""

import pandas as pd
import numpy as np
import seaborn as sns #for a sample dataset


# =============================================================================
# Pandas
# =============================================================================
iris = sns.load_dataset('iris') #load as dataframe

# display options
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)

iris.head()
iris.tail(10)

#check columns, index
iris.columns
iris.index
iris.axes

#counts
iris.shape


iris.dtypes

iris.info() #check null counts, columns, dtypes, memory usage

#check uniques
iris["species"].unique()
iris["species"].nunique()


#check if dataframe is completely empty
iris.empty # NaNs are not considered empty

#selecting columns
iris["sepal_length"]
iris[["sepal_length", "sepal_width"]]

#by index
iris.iloc[:, [0, 4]]
iris.iloc[:, 0:3]

#by label/name
iris.loc[:, ['sepal_length', 'sepal_width']]
iris.loc[:, 'sepal_length':'petal_length']


#describe dataset
iris.describe()
iris.groupby(['species']).describe()
iris.groupby(['species']).describe()["sepal_length"] #choose 1 dim


iris.reset_index() #reset row index


pd.value_counts(iris["species"])

#convert a dataframe/series to numpy (2D/1D arrays for dataframe and series respectively)
iris.to_numpy()

#get top largest/smallest
iris.nlargest(3, columns = ["sepal_length"],
              keep = 'all') #keep all keeps duplicate values


iris.nlargest(3, columns = ["sepal_length", "sepal_width"],
              keep = 'all') #keep all keeps duplicate values

iris.nsmallest(3, columns = ["sepal_length", "sepal_width"])


#select unique/drop dupes
iris[["sepal_length", "species"]].drop_duplicates()
iris.drop_duplicates(subset = ['sepal_length', 'species'])


#check duplicated rows
iris[iris[["sepal_length", "species"]].duplicated(keep = False)] #return all the duplicated rows



# =============================================================================
# Data Types & Casting
# =============================================================================
#Pandas dtypes list:
def subdtypes(dtype):
    subs = dtype.__subclasses__()
    if not subs:
        return dtype
    return [dtype, [subdtypes(dt) for dt in subs]]
subdtypes(np.generic) #list of datatypes and child

#common short string datatypes:
# "int" (can also be specified e.g. np.int32, int64) #default int = int64
# "float" (can also be specified e.g. np.float32, np.float64) #default float = float64
# "object" (string or mixed (be careful storing mixed datasets))
# "string"
# "category"
# "datetime64"

#select columns by dtypes
iris.select_dtypes(include = ['float'], exclude = ['object'])

iris.select_dtypes(include = ['generic'])

#=================
# casting
#=================
#to object (mixed data types)
iris["sepal_length"].astype("object") #not inplace
iris["sepal_length"].astype(str) #also to object

#to string
iris["sepal_length"].astype("string")

#numeric vars (everything works)
iris["sepal_length"].astype("float") #not inplace
iris["sepal_length"].astype(float)
iris["sepal_length"].astype("float32")

iris["sepal_length"].astype("int")
iris["sepal_length"].astype("int64")
iris["sepal_length"].astype(int)


#category dtype
iris["species"].astype("category")


#cast in-place
iris["species"] = iris["species"].astype("string") #casting data types
iris.dtypes

#also (no pd.to_string)
pd.to_numeric(iris["sepal_length"]) #not inplace
pd.to_datetime(['2010-10-01'])
iris["sepal_length"].to_string()



#convert multiple cols using dict
iris.astype({'sepal_length': 'int',
             'species':'string'}).dtypes


# =============================================================================
# Datetimes
# =============================================================================
pd.Series(['2010-10-01']).astype("datetime64")
pd.Series(['2010-10-01']).astype(np.datetime64)
pd.to_datetime(pd.Series(['2010-10-01']))

# create sequence of dates
pd.date_range("2020-01-01", "2020-01-31", periods = 5)
pd.date_range("2020-01-01", "2020-01-31", freq = "3D") # 3-day frequency
pd.date_range("2020-01-01", "2020-01-31", freq = "W") #weekly frequency

#get rid of datetime
pd.Series(pd.date_range("2020-01-01", "2020-01-31", periods = 5)).dt.date
pd.Series(pd.date_range("2020-01-01", "2020-01-31", periods = 5).date)
pd.Series(pd.date_range("2020-01-01", "2020-01-31", periods = 5)).dt.strftime("%Y-%m-%d").astype("datetime64")



# null datetime = NaT
pd.Series(pd.NaT).isnull()


#dt examples
dt_df = pd.DataFrame({"date" : ['2010-10-01', '2010-10-05', '2010-11-01', '2010-12-01', '2011-01-01']}).astype("datetime64")







# Explode to fill in missing dates
pd.Series(['2010-10-01', '2010-10-05', '2010-11-01']).astype("datetime64")





# ==================
# dt methods
# ==================
dt_df["date"].dt.date #get date portion (if datetime is in timestamp)
dt_df["date"].dt.day
dt_df["date"].dt.month
dt_df["date"].dt.year

dt_df["date"].dt.dayofyear
dt_df["date"].dt.dayofweek; dt_df["date"].dt.weekday #0 - monday, Sunday = 6

# floor and ceil pandas has issues for longer frequencies



dt_df["date"].dt.day_name()
dt_df["date"].dt.month_name()


dt_df["date"].dt.daysinmonth #alias: days_in_month

dt_df["date"].dt.is_month_start
dt_df["date"].dt.is_month_end


dt_df["date"].dt.is_year_start
dt_df["date"].dt.is_year_end



# =============================================================================
# Useful pd Series Methods
# ============================================================================

iris["sepal_length"].between(5, 6)

iris["sepal_length"].clip(3, 5) # winsorize (lower, upper)

iris["species"].unique()
iris["species"].nunique()

iris["species"].isin(['setosa','virginica'])

iris.where(iris["sepal_length"] > 5)
iris.where(iris["sepal_length"] > 5, 100) #replace



# ==================
# str methods
# ==================
# most methods are the same as standard python str methods

# string matching
iris["species"].str.contains("set")
iris["species"].str.match("set")
iris["species"].str.fullmatch("set")

# extract w. regex
iris["species"].str.extract(r"([s])") # also extractall (all matches)

# len
iris["species"].str.len()

# index matching
iris["species"].str.index("set")
iris["species"].str.index("set")



# remove white spaces and newlines (\n, \t)
iris["species"].str.strip() #also lstrip() and rstrip()
iris["species"].str.strip("a")


# replace
iris["species"].str.replace("a", # regex
                            "asd").unique()
iris["species"].str.replace(" ", "") # delete all whitespace


# splicing
iris["species"].str.slice(start = 3) # slice first 3 chars, also slice_replace
iris["species"].str.split("t")
iris["species"].str.rsplit("i") # from the right
iris["species"].str.rsplit("i") # from the right
iris["species"].str.partition("i") #splits into 3, left, separator and right (only splits on first occurence)



# check string types
iris["species"].str.isalpha() # also isalnum()
iris["species"].str.isnumeric()
iris["species"].str.isdecimal()



# =============================================================================
# Pandas Filter (To Select Columns/Index)
# =============================================================================
iris.filter(like = 'sepal_', axis = 1) #filter columns 

iris.filter(items = ['sepal_length', 'sepal_width'], axis = 1) #filter columns by actual names (same as loc)

# filter can also be used on named row indexes

# =============================================================================
# New Columns & pd.DataFrame.assign
# =============================================================================
iris["new_col1"] = 1
iris.drop(["new_col1"], axis = 1, inplace = True) # drop columns default axis = 0 and default inplace = False

# not in place
# useful for creating new groups before calling groupby (one-liner)
iris.assign(new_col2 = lambda x : x["sepal_length"] + x.petal_length, # can use lambda functions 
             new_col3 = iris["new_col1"], # reference existing column directly
             new_col4 = iris["sepal_length"] * 3, # if not using lambdas, can reference directly
             new_col5 = lambda x : x.new_col4 + 5 # can reference new column created in assign
    )

#if want to use dict and column names in var use **
new_col_name = "new_col6"
iris.assign(**{new_col_name : lambda x : x.new_col4})



# renaming columns
iris.rename({"species":"species_new"}, inplace = False, axis = 1)

# =============================================================================
# Filtering/Query
# =============================================================================
iris[iris["species"] == "setosa"]

iris[iris["sepal_length"] > 5]

iris[(iris["sepal_length"] > 5) & (iris["petal_length"] > 1)] #multiple categories use parantheses for each condition

# is in method
iris[iris["species"].isin(["setosa",'virginica'])] 


#between
iris[iris["sepal_length"].between(5, 6)]
iris[(iris["sepal_length"] >= 5) & (iris["sepal_length"] <= 6)]

#using query (fast filters)
iris.query('(sepal_length > 5) & (`sepal_length` > sepal_width)') #use backticks for column names with whitespaces ' '

outside_var = iris["sepal_width"]
iris.query('(sepal_length > 5) & (`sepal_length` > @outside_var)') #use @ to refer outside variables

iris.query('(sepal_length + sepal_width) > 5')


#check nulls using query
iris.query('sepal_length == sepal_length') #only include non missing sepal length since NaNs are not equal to itself
#print(np.nan == np.nan) always false
iris.query('sepal_length != sepal_length') #similarly will show missing sepal length
#or
iris.query('sepal_length.isnull()', engine = "python")
iris.query('sepal_length.notnull()', engine = "python")



# math functions available:
# floor, ceil, log, exp, sqrt, ... etc from math module
iris.query('not (sqrt(sepal_length)*2 > 5 or abs(sepal_width) > 2)')  #can use not & and/or

#in/not in
iris.query('species in ("setosa", "virginica")')  #can use not & and/or
iris.query('species not in ("setosa", "virginica")')  #can use not & and/or

# LIKE % filter
# takes on regex
iris[iris["species"].str.startswith('set', na = False, case = True)] #na = False maps NaNs to False (so that NaN doesn't propagate through as NaN)
iris[iris["species"].str.contains('gin', na = False)]
iris[iris["species"].str.endswith('ca', na = False)]
help(pd.Series.str.contains)


# case when statements (use np.where)
iris.assign(new_test_var = np.where(iris["sepal_length"] > 5, "long",
                                    np.where(iris["sepal_length"] > 4, "medium", "short")))


# =============================================================================
# Missing Data
# =============================================================================
iris["species"].isnull()
iris["species"].notnull()

#alias for null
iris["species"].isna()
iris["species"].notna()

#also available in pd methods
pd.isna(iris["species"]) #pd.notna
pd.isnull(iris["species"]) #pd.notnull



#check counts of non-null records
iris[iris["species"].notnull()]["species"].value_counts() 


#drop
iris.dropna()  #drop all rows that have at least 1 NA
iris.dropna(thresh = 2)  #drop all rows that have at least 2 NA columns
iris.dropna(how = 'all') #only drop rows that are ALL NA
iris.dropna(inplace = True) #drop in place


iris.dropna(subset = ['sepal_length', 'petal_length']) #subset looks for which columns to look for NAs


#fillna
iris['sepal_length'].fillna(method = 'ffill', inplace = False) #forward fill (last obs carried to current)
iris['sepal_length'].fillna(method = 'bfill', inplace = False) #back fill (current record carried backwards)



# =============================================================================
# Sorting
# =============================================================================
#sorting by column values
iris.sort_values(by = ['species', 'sepal_length', 'petal_length'],
                 ascending  = [0, 1, 1],
                 inplace = False, #inplace
                 na_position = 'last')


iris.sort_values(by = ['species'],
                 ascending  = [0],
                 ignore_index = True) #if ignore index, the row index returned will be 0, 1, ... n-1 instead of the actual sorted index

#sort based on (default = row) index
iris.sort_index()
iris.sort_index(axis = 0, level = 0) #axis = 0 => row index


# ******************************************************************************
# =============================================================================
# ***** Group By Object *****
# =============================================================================
# ******************************************************************************
'''
https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
Split -> Apply -> Combine
Apply Step: Aggregation/Transformation/Filtration
    Aggregation: compute a summary statistic (or statistics) for each group
    Transformation: perform some group-specific computations
    Filtration: discard some groups, according to a group-wise computation
'''

# returns a GroupBy object 
grouped = iris.assign(dummy_group = lambda x : np.round(x["sepal_width"])).groupby(["species", "dummy_group"]) # 2 groups
type(grouped)


# by default group by sorts the object (asc)
iris.sort_values("species", ascending = 0).groupby(["species"]).mean() # resorted back because of group by
iris.sort_values("species", ascending = 0).groupby(["species"], sort = False).mean() # stays the same since sort = False

# group by NaN group (by default groupby drops NaNs as the grouping)
iris.groupby(["species"], dropna = False).mean() # will show "NaN" as a key group if the key contains NaNs


# groups 
# contains dict of key = group keys and values = row index belonging to that group
grouped.groups
len(grouped) # total # of groups

# selecting groups
grouped.get_group(("setosa", 3.0)) # use tuples for multiple group keys
type(grouped.get_group(("setosa", 3.0))) # returned type is a DataFrame

#check # of uniques
grouped.nunique()


# HAVING/Filter
# returns a DataFrame
iris.groupby(["species"]).filter(lambda x : np.min(x["sepal_length"]) > 4.5)
iris.groupby(["species"]).filter(lambda x : len(x) > 10) # having count(*) > 10

# having -> will exclude the WHOLE group that does not satisfy the criteria (not just records like "WHERE" statement)
iris.query("sepal_length > 5")["species"].value_counts()
iris.query("sepal_length > 5").groupby(["species"]).filter(lambda x : len(x) > 22) # want count(*) per group > 22


#note that np.nan != np.nan (use itself value to get NaN values)
#check non null:
iris.query("sepal_length == sepal_length") #non-null

#check nulls
iris.query("sepal_length != sepal_length") #NULL since np.nan != np.nan



# =============================================================================
# Group By Agg.
# =============================================================================
# can use a function/internal pd.Series method by passing a string (e.g. count, nunique)
# new name = ('variable', function/string method)
# preferred method:
iris.groupby(['species']).agg(mean_petal_length = ('petal_length', np.mean),
                              mean_sepal_length = ('sepal_length', np.mean),
                              sum_petal_length = ('petal_length', np.sum),
                              count = ('petal_length', 'count'), #use pandas predefined function count (exclude NaNs)
                              count_np = ('petal_length', lambda x: np.sum(~np.isnan(x))), #custom functions with numpy (exclude NaNs)
                              size = ('petal_length', np.size), #use numpy size (include NaNs)
                              count_unique = ('petal_length', lambda x: x.nunique()), #using pd.Serires.nunique method
                              count_unique_str = ('petal_length', 'nunique')) #using pandas default function nunique
                              
#more pandas Series internal methods specified using strings
#can pass in a tuple ('new column name', function/string method) to rename the new column
iris.groupby(['species'], as_index = False).agg({'petal_length': [('count_petals', 'count'), 'sum',
                                                                  'mean', 'median',
                                                                  'std', 'var',
                                                                  'min', 'max',
                                                                  'first', 'last']})




#group by as_index = False (prevent multi-index/hierarchical)
iris.groupby(['species'], as_index = False)[['petal_length', 'sepal_length']].mean()

#Agg with dictionary (returns multi column index)
grouped = iris.groupby(['species'], as_index = False).agg({'petal_length': [np.mean, np.sum],
                                                 'sepal_width': [np.sum, 'count']})
print(grouped)
grouped.columns # multiindex


#flatten
grouped.columns  = ['_'.join(col).strip() for col in grouped.columns.values] #recommended
#or
# grouped.columns  = grouped.columns.get_level_values(0)  #only takes the level values -> duplicate names
#or
# grouped.columns = grouped.columns.to_flat_index() #('petal_length', 'mean') as column names (not recommended)


# =============================================================================
# Transformation**
# =============================================================================
# transform returns DataFrame with the SAME SIZE/records with the original DataFrame (very useful)
# useful for substracting mean for EACH GROUP etc.
# transform processes the columns series by series (hence, cannot do lambda x : x.col, since x is a series (contrast, w. apply))
iris.groupby(['species'])["sepal_length"].transform(lambda x : x - np.mean(x))

# returns mean of each group for ALL records (& all columns)
# cannot use dict to specify the column since transform passes series
iris.groupby(['species']).transform(lambda x : np.mean(x)) 

# returns the min of EACH group for ALL records (not just displaying the 3 records)
iris.groupby(['species'], as_index = False).transform(lambda x : np.min(x))
iris.groupby(['species'], as_index = False).apply(lambda x : np.min(x)) # contrast to this


# select * from ... where (select max(...) from ... group by ...)
iris[iris["sepal_length"] == iris.groupby(['species'])["sepal_length"].transform(np.max)]



# =============================================================================
# Apply
# =============================================================================
# most flexible, can be used as a transformer / aggregator (also the slowest)
# apply passes through a DataFrame (hence, possible to do lambda x : x.col, since x is a DataFrame)

# as transformer:
iris.groupby(['species'], as_index = False).apply(lambda x : x.petal_length - np.mean(x.petal_length)) 


# as aggregator
iris.groupby(['species'], as_index = False).apply(lambda x : np.mean(x.petal_length)) 


# in apply the passed function: lambda x, x is a DataFrame, hence can do referencing such as x.sepal_length
iris.groupby(['species'], as_index = False).apply(lambda x : x.sepal_length - np.mean(x["sepal_length"]))

# can also use describe
iris.groupby(['species']).apply(lambda x : x.describe())



# can use display to visualize
# def substract_mean(x):
#     display(x)
#     return x['sepal_length'] - np.mean(x['sepal_length'])

iris.groupby(['species']).apply(substract_mean)

# =============================================================================
# Useful Common (Window) Functions (By Group)
# =============================================================================
# ===================
# Cumulative Sum
# ===================
iris.sort_values('sepal_length').groupby('species', sort = False).apply(lambda x : np.cumsum(x.sepal_length)).reset_index() #using apply
iris.sort_values('sepal_length').groupby('species', sort = False)['sepal_length'].transform(lambda x : np.cumsum(x)) #using transform


# ===================
# Row Number
# ===================
iris.sort_values("sepal_length").assign(rn = np.arange(len(iris)) + 1)
iris.sort_values("sepal_length").assign(rn = lambda x : np.cumsum(x.sepal_length == x.sepal_length))


# ===================
# Cumulative count by group
# ===================
iris.assign(rn = iris.sort_values("sepal_length", ascending = False).groupby('species').cumcount() + 1).sort_values("rn") 

iris.assign(rn = iris.sort_values("sepal_length", ascending = False).groupby('species', as_index = False) \
            .apply(lambda x : np.cumsum(x.sepal_length == x.sepal_length))).sort_values("rn") 

iris.assign(rn = iris.sort_values("sepal_length", ascending = False).groupby('species')['sepal_length'] \
            .transform(lambda x : np.cumsum(x == x))).sort_values("rn") 

    
pd.concat([iris, iris.groupby('species')['sepal_length'].transform(lambda x : np.cumsum(x == x))], axis = 1)


# ===================
# Max By Group
# ===================
iris.groupby('species')['sepal_length'].transform(lambda x : np.max(x))


# ===================
# Lead/Lag (Window w. Transform)
# ===================
iris.assign(lead_val = #lead
            iris.sort_values(['species', 'sepal_length'])
                .groupby('species')['sepal_length'].shift(periods = 2, fill_value = None) # lag use negative value
           ) #no need to sort before assigning, index will take care of it

# ===================
# First/Last (Window w. Transform)
# ===================
iris.assign(first = 
            iris.sort_values(['species', 'sepal_length']) \
                .groupby('species')['sepal_length'].transform(lambda x : np.repeat(x.head(1), len(x)))
            )
           

iris.assign(last = 
            iris.sort_values(['species', 'sepal_length']) \
                .groupby('species')['sepal_length'].transform(lambda x : np.repeat(x.tail(1), len(x)))
            )

# put first 3 in a list           
iris.assign(first_3 = 
                iris.sort_values(['species', 'sepal_length']) \
                    .groupby('species')['sepal_length'].transform(lambda x : [x.head(3).to_numpy() for i in range(len(x))] )
            ) # can be used next for .explode()


#lead w. transform
iris.groupby('species')['sepal_length'].transform(lambda x : x.shift(1)) #can use shift/roll here
    
    
# =============================================================================
# Windowing
# =============================================================================
'''
https://pandas.pydata.org/pandas-docs/stable/user_guide/window.html
'''
# Rolling
iris.sort_values(['sepal_length']).rolling(window = 5,
                                           min_periods = 3,
                                           center = False).apply(lambda x : np.sum(x.head(4))  #include first 4
                                                                 )

iris.rolling(5).sum()
iris.rolling(5).std()


# forward rolling
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=3)
iris.rolling(window=indexer, min_periods=3).sum() #sum current row -> current row + 2


# Expanding
iris.expanding(min_periods = 3, center = None).sum()
iris.expanding(min_periods = 3, center = True).sum()

# Exponential weighted (ewm)


# custom window Indexer
# api.indexers.FixedForwardWindowIndexer([…]) -> fixed-length forward looking windows that include the current row.
# api.indexers.BaseIndexer -> Base class for window bounds calculations.
# api.indexers.VariableOffsetWindowIndexer([…]) -> Calculate window boundaries based on a non-fixed offset such as a BusinessDay



# =============================================================================
# Iterating through GroupBy Object
# =============================================================================





# ******************************************************************************
# ******************************************************************************



# =============================================================================
# Joins/Merge
# =============================================================================
# merges keep only one column on key columns
# merge
pd.merge(left = iris, right = iris,
         how = 'inner', # can use 'cross' to create cartesian product
         left_on = ['sepal_length','sepal_width'],
         right_on = ['sepal_length','sepal_width'],
         suffixes = ("_x", "_y"), #or none, to indicate same column names after merging
         indicator = True, #shows the source of the column (from both tables, left_only or right_only)
         validate = 'many_to_many') #can validate joins if it's many_to_many, one_to_one, etc...


#check column differences
iris.columns.difference(iris.columns)



#join based on row index
pd.merge(left = iris, right = iris,
         left_index = True,
         right_index = True) 


# unequi joins
# join then query
pd.merge(left = iris, right = iris, left_on = 'sepal_length',
         right_on = 'sepal_width', how = 'left', indicator = True).query("sepal_width_y > 2")

#or merge_asof
# pd.merge_asof




# =============================================================================
# MultiIndex/Hierarchical Index
# =============================================================================





# =============================================================================
# Useful pd functions
# =============================================================================
pd.cut
pd.qcut




# =============================================================================
# Pivot/Melt
# =============================================================================
# Long to Wide
pd.pivot(iris, columns = ["species"])


# Wide to Long
# pd.melt(iris, id_vars = )







