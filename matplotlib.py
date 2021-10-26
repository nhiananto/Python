# =============================================================================
# MATPLOTLIB Reference
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(10) + 2
y = np.random.randn(10) + 2

fig, ax = plt.subplots()

#plot dots
ax.plot(x, y,
        lw = 0, #have to specify linewidth
        marker = 'o', #'o' #'.' #',' #'^', #'<' #'>', 'v' etc
        markersize  = 5, #ms for short
        markeredgecolor = 'black', #mec 
        markeredgewidth = 0.75, #mew
        markerfacecolor = 'orange', #mfc
        zorder= 1, #zorder sets draw order, lower zorder are drawn first
        label = 'dots') #automatically create legend by calling ax.legend() (no args)

#or use fmt argument string "o", fmt = '[marker][line][color]'
ax.plot(x+2, y+2, "or", markersize = 1, label ='dots fmt \n string') #fmt = 'or' = red dots

#or use scatter
ax.scatter(x, y, color = 'blue',
           s = 100, #size
           marker = '^', alpha = 0.5,
           edgecolor = 'black',
           lw = 2) #note that edge is affected by alpha

#can use RGB values so that edge is not affected by alpha
ax.plot(x+1, y+1,
        lw = 0, #have to specify linewidth
        marker = 'o', #'o' #'.' #',' #'^', #'<' #'>', 'v' etc
        ms  = 7,
        mew= 1,
        markeredgewidth = 2, #mew
        mfc = (1,0,0,0.3), #last value is alpha
        mec = (0,0,1, 1), #last value is alpha, 3values is RGBA (0 - 1)
        label = 'dots alpha edge')

#can use colors to convert common color names to RGBA values
import matplotlib
red_RGBA = matplotlib.colors.to_rgba('red')

#create lines
ax.plot([1, 3],
        [0, 4],
        c = 'black', #color
        label = 'line1',#label used for legends
        lw = 0.75, #linewidth
        ls = '-')  #linestyle

#set x y lim
ax.set_xlim([0,5])
ax.set_ylim([0,7])
#or
#quick axis label setup properties
ax.set(xlim = [0,5], ylim = (0,5), xlabel = "asd", ylabel = "y") 

ax.set_xlabel(xlabel = 'time \n (s)',
              fontsize = 15,
              fontweight = 'bold', #0-1000 or normal bold etc
              horizontalalignment = 'left',
              wrap = False,
              labelpad = -10, #distance from inner plot
              color = "blue",
              alpha = 0.75,
              rotation = 15,
              position = [1,10000] #y value does not matter set through labelpad
              )
ax.set_ylabel('$y_2 = \epsilon$', position = [10000, 1]) #accepts latex

#title
ax.set_title('PLOT OOP',
             loc = 'left', #left center right
             pad = 10,
             fontsize = 16)


#or
ax.set_title('PLOT OOP Right',
             position = [0.75,1],
             fontsize = 16)

#suptitle
fig.suptitle("What a title", x = 0.25, y = 1, fontsize = 20)

#vertical lines
ax.vlines([1,3],
          ymin = 0,
          ymax = 5,
          linestyles = ['solid','dashed']) #linestyles or linestyle or ls

#horizontal lines
ax.hlines([1,3],
          xmin = ax.get_xbound()[0],
          xmax = ax.get_xlim()[1],
          ls = ['dashdot','dashed'],
          color = 'red',
          lw= [0.5, 2]) #linewidth, linewidths, or lw

#axis to axis hline and vline
ax.axvline(2.5, ymin = 0, ymax = 0.85, color = 'blue', ls = 'dashed', lw = 2) #from axis to axis
ax.axhline(2, xmin = 0.2, ls = (0.5, (1,1.5)) )#set your own dashed type with the dash tuple

#polygons collection
ax.fill_betweenx(y = [1,3], x1 = 2.5, x2 = 3, alpha = 0.5, edgecolor = 'black', lw = 5) #no mec argument
ax.fill([0.2,0.3,0.7,0.9,0.2], [1,2,2.5,1.5,1], fc= (0,0,0,0), ec = (1,0,0,1), lw = 3)
#also see ax.fill_between

#axis rectangles
ax.axvspan

#call legend (with no arguments) will automatically create legend with the designated labels
ax.legend()


# =============================================================================
# Histograms
# =============================================================================
# example data
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
x = mu + sigma * np.random.randn(437)

num_bins = 50

fig, ax = plt.subplots()
ax.hist(x, bins = num_bins,
        density=True,
        align = 'mid',
        facecolor = 'magenta',
        edgecolor = 'black')
# =============================================================================
# Legends (if label arguments are not used)
# =============================================================================
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# Creating explicit legends with proxy artists
# create 4 different legends/proxy artists
red_patch = mpatches.Patch(color='red', label='The red data')

blue_line = mlines.Line2D([], [], color='blue', marker='*',
                          markersize=15, label='Blue stars')

red_dots = mlines.Line2D([],[], lw = 0, c = 'red', marker = 'o', ms = 5, label = 'The Red Dot')

line_markers = mlines.Line2D([],[], lw = 2, c = 'red', marker = 'o', mfc = 'blue', mec = 'black',  ms = 5, label = 'The blue dot marker \n black edge')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.legend(handles=[red_patch, blue_line, red_dots, line_markers])

# ===================================
# legend locations and formats
# ===================================
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.legend(handles=[red_patch, blue_line, red_dots, line_markers],
          loc= (0,0), #x,y coordinates of bottom left legend box
          title = 'Legend',
          title_fontsize = 15,
          prop = {'size':12,
                  'weight':600,
                  'style':'italic'},
          facecolor = 'grey', #background
          framealpha = 0.5, #alpha of background color
          edgecolor = 'red',
          borderpad = 2, #padding around border
          labelspacing = 1, #vertical spacing among legends
          columnspacing = 1, #column spacing
          handlelength = 1, #length of logo/handles
          handletextpad = 0.5, #padding between handle and the text
          ncol = 2)


#more on bbox
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.legend(handles = [red_patch], 
          bbox_to_anchor=(0., 1.02, 1., .102),  #4-tuple (x, y, width, height)
          loc='lower left', #location of starting bbox
          borderaxespad=0. #padding between axes border and legend)


#legend loc
# Location String	Location Code
# 'best'	0
# 'upper right'	1
# 'upper left'	2
# 'lower left'	3
# 'lower right'	4
# 'right'	5
# 'center left'	6
# 'center right'	7
# 'lower center'	8
# 'upper center'	9
# 'center'	10

# ==========================
# multiple legends
# (have to add manually to axes) since new legend() overwrites old ones
# ==========================
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

# 1st legend
first_legend = ax.legend(handles=[red_patch], loc='upper right')
#add to axis
ax.add_artist(first_legend)
#second legend
ax.legend(handles = [red_patch], loc = 'lower left')

# =============================================================================
# Text and Annotations
# =============================================================================
fig = plt.figure()
ax = fig.gca()
ax.set(xlim = [0,5], ylim = [0,5])


ax.text(3, 3, 'boxed text in \n data coords',
        fontstyle='italic',
        fontweight = 900,
        fontsize = 8,
        rotation = 10,
        color = 'blue',
        horizontalalignment = 'left',
        bbox={'facecolor': 'grey',
              'edgecolor':'red',
              'boxstyle':'round',
              'ls':'--',
              'lw':2,
              'alpha': 0.5,
              'pad': 2})
#additional bbox reference:
#https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.patches.FancyBboxPatch.html#matplotlib.patches.FancyBboxPatch
#https://matplotlib.org/3.3.0/gallery/shapes_and_collections/fancybox_demo.html#sphx-glr-gallery-shapes-and-collections-fancybox-demo-py


ax.text(0.2, 0.5, 'colored text \n in axes coords',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)


#arrows and annotations
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.arrow(1,1,2,2, head_width = 0.2) #scaled with x axis, non "square" arrow

# =============================================================================
# Subplots and Layouts and Margins
# =============================================================================


# =============================================================================
# Tick labels and axis
# =============================================================================
#turn off all axis x& y
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_axis_off()

#turning off single axis
fig = plt.figure()
ax = plt.gca() #get current axes
ax.spines['bottom'].set_visible(False)
ax.get_xaxis().set_visible(False) #turn off labels

#invert axis
fig = plt.figure()
ax = plt.gca() #get current axes
ax.invert_yaxis()
#set x boundaries 
#similar to xlim but changes automatically if you include data that is outside the boundary
ax.set_xbound(0, 1) 

#ticks and major ticks
fig = plt.figure()
ax = plt.gca() #get current axes
ax.set_xticks([0,0.5,1.0])
ax.set_xticklabels(['a','b','c'],
                   color = 'blue',
                   alpha = 0.75,
                   fontsize = 13,
                   fontweight = 100,
                   fontstyle = 'italic',
                   rotation = 90)
                   # position = [9999, -0.05]) #x value does not matter
                   # better to set on pad on tick_params
ax.set_xticks([0.1,0.2,0.7,0.8], minor = True) #set minor
ax.set_xticklabels([0.1,0.2,0.7,0.8], minor = True)

#grid lines
ax.grid(True, which = 'both') #which shows major and minor
#control grid properties independently
ax.xaxis.grid(True, which = 'major',color = 'r', ls = '--', lw = 2) 
ax.xaxis.grid(True, which = 'minor',color = 'blue', ls = '--', lw = 1)
ax.yaxis.grid(True, which = 'major', color = 'black', ls = '--')

#set tick lines formatting
ax.tick_params(axis = 'x',
               which = 'major',
               direction = 'inout', #in, out, inout
               length = 15, #length of lines/ticks
               width = 2,
               color = 'red',
               pad = 5, #distance between tick and label can also be set by position on labels
               top = True, #also draw at the top
               labeltop = True, #draw labels at top too
               labelcolor = 'red', #can also set label properties
               ) 


#change decimals/format numbers
from matplotlib import ticker
formatter = ticker.StrMethodFormatter('{x:1.3f}')
ax.xaxis.set_minor_formatter(formatter)

#getting xlabels
ax.get_xticklabels(minor = False)
ax.get_xticklabels(minor = True)
ax.get_xminorticklabels()
ax.get_xmajorticklabels()

#remove minor ticks
# ax.minorticks_off()


# =============================================================================
# Color Cyclers
# =============================================================================
from cycler import cycler #set cycler

#create data
x = np.linspace(0,1, 10).reshape(-1,1)
x = np.tile(x, reps = 10)
y = np.repeat(np.arange(10), 10).reshape(10,10, order = 'F')
fig, ax = plt.subplots()
#cycle list of RGBA values of gray colors
ax.set_prop_cycle(cycler(color = [(x, x, x, 1) for x in np.linspace(0, 0.8, 10)][::-1]))
ax.plot(x, y)


# =============================================================================
# Fill between example
# =============================================================================
y = np.arange(0.0, 2, 0.01)
x1 = np.sin(2 * np.pi * y)
x2 = 1.2 * np.sin(4 * np.pi * y)

fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharey=True, figsize=(6, 6))

ax1.plot(x1, y, c = 'black')
ax1.fill_betweenx(y, 0, x1)
ax1.set_title('between (x1, 0)')

ax2.plot(x1, y, c = 'black')
ax2.fill_betweenx(y, x1, 1)
ax2.set_title('between (x1, 1)')
ax2.set_xlabel('x')

ax3.plot(x1, y, x2, y, color='black')
ax3.fill_betweenx(y, x1, x2)
ax3.set_title('between (x1, x2)')

def rand_data():
    return np.random.uniform(low=0., high=1., size=(100,))

# Generate data.
x1, y1 = [rand_data() for i in range(2)]
x2, y2 = [rand_data() for i in range(2)]
