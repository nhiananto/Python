# =============================================================================
# Random Variables Reference
# =============================================================================
from numpy import random
from scipy import stats


#https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html
#https://stackoverflow.com/questions/36391970/what-does-see-docstring-of-the-instance-object-for-more-information-mean

# =============================================================================
# normal distribution
# =============================================================================
stats.norm.pdf(0.5) #pdf
stats.norm.ppf(0.5) #inverse cdf given quantile
stats.norm.cdf(0) #cdf
stats.norm.cdf(0, loc = 1, scale = 2) #loc = mean, scale = std
#generate normal
stats.norm.rvs(size = (5,3))

# Alternatively, stats.norm is also callable -- you can pass the loc and scale "shape" parameters to "freeze" those parameters into the distribution. What you get back is called a "frozen distribution"
x = stats.norm(10,2)
x.rvs(size = 10)

#numpy
random.normal(0, 1, 10)
random.standard_normal(10)


# =============================================================================
# uniform
# =============================================================================
stats.uniform.rvs(0, 1, size = 10)
#numpy
a = random.uniform(0, 1, (10,2))

# =============================================================================
# t-dist
# =============================================================================
stats.t.ppf(0.25, df = 3)
stats.t.cdf(0.25, df = 3)

stats.t.rvs(df = 5, size = 10)

#numpy
random.standard_t(df = 5, size = 10)
