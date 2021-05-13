from __future__ import print_function
import sys, os
import numpy as np
import cv2
import h5py
from copy import copy
import matplotlib.pyplot as plt

os.chdir(r'D:\Documents\Akademia\Aktuell\MA_IGS\models\output')

# visualize history

# load data
model_name = 'tEDRAM, 5 steps'
folder_name = 'history'
reps = 1

if not os.path.exists('images'):
            os.makedirs('images')

acc = []
val_acc = []
cla_loss = []
val_cla_loss = []
loc_loss = []
val_loc_loss = []
dim_loss = []
val_dim_loss = []
lr = []

for i in range(1, reps+1):

    acc.append(np.load(folder_name + str(i) + '/classifications_categorical_accuracy.npy'))
    val_acc.append(np.load(folder_name + str(i) + '/val_classifications_categorical_accuracy.npy'))
    cla_loss.append(np.load(folder_name + str(i) + '/classifications_loss.npy'))
    val_cla_loss.append(np.load(folder_name + str(i) + '/val_classifications_loss.npy'))

    loc_loss.append(np.load(folder_name + str(i) + '/localisations_loss.npy'))
    val_loc_loss.append(np.load(folder_name + str(i) + '/val_localisations_loss.npy'))

    dim_loss.append(np.load(folder_name + str(i) + '/dimensions_loss.npy'))
    val_dim_loss.append(np.load(folder_name + str(i) + '/val_dimensions_loss.npy'))

    lr.append(np.load(folder_name + str(i) + '/lr.npy'))

acc = np.asarray(acc)
val_acc = np.asarray(val_acc)
cla_loss = np.asarray(cla_loss)
val_cla_loss = np.asarray(val_cla_loss)

loc_loss = np.asarray(loc_loss)
val_loc_loss = np.asarray(val_loc_loss)

dim_loss = np.asarray(dim_loss)
val_dim_loss = np.asarray(val_dim_loss)

lr = np.asarray(lr)

print(np.max(val_acc, axis=1),"\n", np.min(val_cla_loss, axis=1),"\n", np.min(val_loc_loss, axis=1))


# plot data
e = 30
x = np.linspace(0,e-1,e)

# classification loss
best = np.min(cla_loss, axis=0)
mean = np.median(cla_loss, axis=0)
last = np.max(cla_loss, axis=0)
plt.plot(mean[:e], color='blue')
#plt.fill_between(x, best[:e], last[:e], facecolor='red', color='blue', alpha=0.3)

best = np.min(val_cla_loss, axis=0)
mean = np.median(val_cla_loss, axis=0)
last = np.max(val_cla_loss, axis=0)
plt.plot(mean[:e], color='orange')
plt.ylim([0,6])
#plt.fill_between(x, best[:e], last[:e], facecolor='red', color='orange', alpha=0.3)

plt.title(model_name+' Classification Loss')
plt.ylabel('Categorical Cross-Entropy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(fname='images/cla_loss.png', dpi=150)


# localisation loss
best = np.min(loc_loss, axis=0)
mean = np.median(loc_loss, axis=0)
last = np.max(loc_loss, axis=0)
plt.plot(mean[:e], color='blue')
#plt.fill_between(x, best[:e], last[:e], facecolor='red', color='blue', alpha=0.1)

best = np.min(val_loc_loss, axis=0)
mean = np.median(val_loc_loss, axis=0)
last = np.max(val_loc_loss, axis=0)
plt.plot(mean[:e], color='orange')
#plt.ylim([0,0.14])
#plt.fill_between(x, best[:e], last[:e], facecolor='red', color='orange', alpha=0.1)

plt.title(model_name+' Localization Loss')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(fname='images/loc_loss.png', dpi=150)


# dimesnion loss
best = np.min(dim_loss, axis=0)
mean = np.median(dim_loss, axis=0)
last = np.max(dim_loss, axis=0)
plt.plot(mean[:e], color='blue')
#plt.fill_between(x, best[:e], last[:e], facecolor='red', color='blue', alpha=0.1)

best = np.min(val_dim_loss, axis=0)
mean = np.median(val_dim_loss, axis=0)
last = np.max(val_dim_loss, axis=0)
plt.plot(mean[:e], color='orange')
#plt.ylim([0,0.14])
#plt.fill_between(x, best[:e], last[:e], facecolor='red', color='orange', alpha=0.1)

plt.title(model_name+' Dimension Loss')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(fname='images/loc_loss.png', dpi=150)


plt.plot(lr[:40].T)
plt.ylabel('learning rate')
plt.xlabel('epoch')
plt.savefig(fname='images/lr.png')

###############################################################################

# visualize search

# load data
f = h5py.File('predictions.h5', 'r')

X = f['features']
Y_lab = f['labels']
Y_loc = f['locations']

edram = True
last_loc = 0
clip = 30

if not os.path.exists('images/perfect'):
            os.makedirs('images/perfect')
if not os.path.exists('images/correct'):
            os.makedirs('images/correct')
if not os.path.exists('images/false'):
            os.makedirs('images/false')

# batch size
n = Y_loc.shape[0] - Y_lab.shape[0] if edram else X.shape[0]//2
# steps
steps = Y_lab.shape[0]//n-1 if edram else 1

emotions = False
if Y_lab.shape[1]==7:
    emotions = True
    labels = ['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger']


# create plots
for i in range(0, n):

    perfect = True
    var = np.zeros((7))

    # true training data
    image = copy(cv2.cvtColor(X[i,:,:,0], cv2.COLOR_GRAY2BGR))
    bb_t = Y_loc[i,:]
    lab_t = np.argmax(Y_lab[i,:])

    if not emotions:
        # draw true bb
        w = int(bb_t[0]*100)
        h = int(bb_t[4]*100)
        if w > clip: w = clip
        if h > clip: h = clip
        if w < 1: w = 1
        if h < 1: h = 1
        x = int(bb_t[2]*50+50)-w//2
        y = int(bb_t[5]*50+50)-h//2
        if x<0: x = 0
        if y<0: y = 0
        if x+w>100: w = 99-x
        if y+h>100: h = 99-y

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # output of the initialization network
    bb_o = Y_loc[i+n,:]
    lab_o = np.argmax(Y_lab[i+n,:])
    var[lab_o] = 1

    if lab_o!=lab_t:
        perfect = False

    # draw prediction
    w = int(bb_o[0]*100)
    h = int(bb_o[4]*100)
    if w > clip: w = clip
    if h > clip: h = clip
    if w < 1: w = 1
    if h < 1: h = 1
    x = int(bb_o[2]*50+50)-w//2
    y = int(bb_o[5]*50+50)-h//2
    if x<0: x = 0
    if y<0: y = 0
    if x+w>100: w = 99-x
    if y+h>100: h = 99-y

    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)

    blank = np.zeros((100,20,3))+255
    output = np.zeros((100,20,3))+255
    cv2.putText(output,str(lab_t), (3,18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    lab = np.zeros((100,20,3))+255
    cv2.putText(lab,str(lab_o), (3,18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128,128,128), 2)
    output = np.hstack([output, image, lab])
    if x<0: x = 0
    if y<0: y = 0
    image = copy(cv2.cvtColor(X[i,:,:,0], cv2.COLOR_GRAY2BGR))
    image = cv2.resize(cv2.resize(image[y:y+h,x:x+w], (26,26), interpolation=cv2.INTER_LINEAR), (100,100), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(image, (0, 0), (99, 99), (0, 0, 255), 2)
    output2 = np.hstack([blank, image, blank])

    for step in range(1,steps+last_loc):

        image = copy(cv2.cvtColor(X[i,:,:,0], cv2.COLOR_GRAY2BGR))

        if not emotions:
            # draw true bb
            w = int(bb_t[0]*100)
            h = int(bb_t[4]*100)
            if w > clip: w = clip
            if h > clip: h = clip
            if w < 1: w = 1
            if h < 1: h = 1
            x = int(bb_t[2]*50+50)-w//2
            y = int(bb_t[5]*50+50)-h//2
            if x<0: x = 0
            if y<0: y = 0
            if x+w>100: w = 99-x
            if y+h>100: h = 99-y

            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

        # output
        bb_o = Y_loc[i+n*(step+1),:]
        if step<steps:
            lab_o = np.argmax(Y_lab[i+n*(step+1),:])
            var[lab_o] = 1

            if lab_o!=lab_t:
                perfect = False

        w = int(bb_o[0]*100)
        h = int(bb_o[4]*100)
        if w > clip: w = clip
        if h > clip: h = clip
        if w < 1: w = 1
        if h < 1: h = 1
        x = int(bb_o[2]*50+50)-w//2
        y = int(bb_o[5]*50+50)-h//2
        if x<0: x = 0
        if y<0: y = 0
        if x+w>100: w = 99-x
        if y+h>100: h = 99-y

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)

        lab = np.zeros((100,20,3))+255
        if step<steps:
            cv2.putText(lab,str(lab_o), (3,18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128,128,128), 2)
        output = np.hstack([output, image, lab])
        image = copy(cv2.cvtColor(X[i,:,:,0], cv2.COLOR_GRAY2BGR))
        image = cv2.resize(cv2.resize(image[y:y+h,x:x+w], (26,26), interpolation=cv2.INTER_LINEAR), (100,100), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(image, (0, 0), (99, 99), (0, 0, 255), 2)
        output2 = np.hstack([output2, image, blank])

    # save plot
    if perfect:
        folder = 'perfect'
    elif lab_o==lab_t:
        folder = 'correct'
    else:
        folder = 'false'
    cv2.imwrite('images/'+folder+'/'+str(sum(var))+'output_'+str(i)+'.png', output)
    cv2.imwrite('images/'+folder+'/'+str(sum(var))+'output_'+str(i)+'_r.png', output2)
    output3 = np.vstack([output, np.zeros((5,(steps+last_loc)*100+(steps+last_loc+1)*20,3))+255, output2])
    for j in range(0,steps+last_loc):
        bb_o = Y_loc[i+n*(j+1),:]
        w = int(bb_o[0]*100)
        h = int(bb_o[4]*100)
        if w > clip: w = clip
        if h > clip: h = clip
        if w < 1: w = 1
        if h < 1: h = 1
        x = int(bb_o[2]*50+50)-w//2
        y = int(bb_o[5]*50+50)-h//2
        if x<0: x = 0
        if y<0: y = 0
        if x+w>100: w = 99-x
        if y+h>100: h = 99-y
        cv2.line(output3, (20+j*(120)+x, y+h), (20+j*(120), 105), (0, 0, 255), 1)
        cv2.line(output3, (20+j*(120)+x+w, y+h), (119+j*(120), 105), (0, 0, 255), 1)
    cv2.imwrite('images/'+folder+'/'+str(sum(var))+'output_'+str(i)+'_z.png', output3)


for label in labels:
    if not os.path.exists('images/emotions/'+label): os.makedirs('images/emotions/'+label)

predicted = True
last_loc = 1

# create plots

heatmap = np.zeros((7,7,100,100))

for i in range(0, n):

    lab_t = np.argmax(Y_lab[i,:])

    if last_loc:

        lab_o = np.argmax(Y_lab[i+n,:])

        bb_o = Y_loc[i+n,:]

        w = int(bb_o[0]*100)
        h = int(bb_o[4]*100)
        if w > clip: w = clip
        if h > clip: h = clip
        if w < 1: w = 1
        if h < 1: h = 1
        x = int(bb_o[2]*50+50)-w//2
        y = int(bb_o[5]*50+50)-h//2
        if x<0: x = 0
        if y<0: y = 0
        if x+w>100: w = 99-x
        if y+h>100: h = 99-y

        for e, l in enumerate(labels):

          if e == lab_o if predicted else lab_t:

            heatmap[e,0,y:y+h,x:x+w] = heatmap[e,0,y:y+h,x:x+w] + 1

    for step in range(last_loc, steps+last_loc):

        if step<steps:
            lab_o = np.argmax(Y_lab[i+n*(step+1-last_loc),:])

        bb_o = Y_loc[i+n*(step+1),:]

        w = int(bb_o[0]*100)
        h = int(bb_o[4]*100)
        if w > clip: w = clip
        if h > clip: h = clip
        if w < 1: w = 1
        if h < 1: h = 1
        x = int(bb_o[2]*50+50)-w//2
        y = int(bb_o[5]*50+50)-h//2
        if x<0: x = 0
        if y<0: y = 0
        if x+w>100: w = 99-x
        if y+h>100: h = 99-y

        for e, l in enumerate(labels):

          if e == lab_o if predicted else lab_t:

            heatmap[e,step,y:y+h,x:x+w] = heatmap[e,step,y:y+h,x:x+w] + 1


for step in range(0,steps+last_loc):

    for e, l in enumerate(labels):

        heatmap[e,step,:,:] = heatmap[e,step,:,:] / (np.max(heatmap[e,step,:,:]) + 0.00000001)
        cv2.imwrite('images/emotions/'+l+'/heatmap_'+str(step)+'.png', heatmap[e,step,:,:]*255)

plot = np.vstack(np.hstack(heatmap[e,step,:,:] for e, l in enumerate(labels)) for step in range(0,steps+last_loc))

template = np.ones((110,102))
plot = np.zeros((0,102*len(labels)))
for e, l in enumerate(labels):
    row = np.zeros((110,0))
    for step in range(0,steps+last_loc):
        h = copy(template)
        h[:100,0:100] = np.mean(X[np.asarray(np.argmax(Y_lab[:n,:], axis=1)==e) * np.asarray(np.argmax(Y_lab[n*step:n*(step+1),:], axis=1)==e),:,:,0], axis=0)/512+heatmap[e,step,:,:]/2
        h[46:54,48:52] = 0
        h[48:52,46:54] = 0
        h[47:53,49:51] = 1
        h[49:51,47:53] = 1
        row = np.hstack([row, h])

    plot = np.vstack([plot, row])

plot = plot[:110*(steps+last_loc)-10,:102*len(labels)-2]
cv2.imwrite('images/heatmap.png', plot*255)
plot = cv2.imread('images/heatmap.png')
plot_col = cv2.applyColorMap(plot, cv2.COLORMAP_OCEAN, i)
cv2.imwrite('images/heatmap_col.png', plot_col)


# Plot heatmap
plt.clf()
plt.title('Heatmap')
plt.imshow(plot, cmap = 'magma')
plt.show()




h=np.mean(X[np.argmax(Y_lab[:n,:], axis=1)==0,:,:,0], axis=0)/1+heatmap[0,6,:,:]*256
h[46:54,48:52] = 0
h[48:52,46:54] = 0
h[47:53,49:51] = 255
h[49:51,47:53] = 255
plt.imshow(h)




    # in case
    x = input()
    if x=='q':
       break