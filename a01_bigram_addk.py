#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:26:24 2018

@author: chaikesh
"""

from nltk.corpus import gutenberg
from nltk.corpus import brown
import numpy as np


# In[2]:

# import sentence in corpus

sent_g=gutenberg.sents()
sent_b=brown.sents()


# In[3]:

# add start and end token to each sentence

sent_gg=[]
sent_bb=[]
for sent in sent_g:
    sent_gg.append([unicode('<s>','utf-8'),unicode('<s>','utf-8')]+[x.lower() for x in sent]+[unicode('</s>','utf-8')])

for sent in sent_b:
    sent_bb.append([unicode('<s>','utf-8'),unicode('<s>','utf-8')]+[x.lower() for x in sent]+[unicode('</s>','utf-8')])


# In[15]:

# train ,test split

train_gl=int(len(sent_g)*0.8)
test_gl =int(len(sent_g)-train_gl)
train_g = sent_gg[0:train_gl]
test_g  = sent_gg[train_gl:]

train_bl=int(len(sent_b)*0.8)
test_bl =int(len(sent_b)-train_bl)
train_b = sent_bb[0:train_bl]
test_b  = sent_bb[train_bl:]
#len(train_g)+len(train_b)
train_bg=train_b+train_g


# In[16]:

# unigram with frequency

train_g_u = {}
for sent in train_g:
    for word in sent:
        key=word
        if key in train_g_u:
            train_g_u[key]=train_g_u[key] + 1
        else:
            train_g_u[key]=1
            
voc_g=len(train_g_u)

            
train_b_u = {}
for sent in train_b:
    for word in sent:
        key=word
        if key in train_b_u:
            train_b_u[key]=train_b_u[key] + 1
        else:
            train_b_u[key]=1

voc_b=len(train_b_u)

train_bg_u = {}
for sent in train_bg:
    for word in sent:
        key=word
        if key in train_bg_u:
            train_bg_u[key]=train_bg_u[key] + 1
        else:
            train_bg_u[key]=1

voc_bg=len(train_bg_u)

#print voc_g,voc_b,voc_bg


# In[17]:

# unigram replacing with "UNK"
th=6
train_g_u['UNK']=0
for key in train_g_u.keys():
    if (train_g_u[key]<= th):
        train_g_u['UNK']+=train_g_u[key]
        del train_g_u[key]  

voc_g=len(train_g_u)

train_b_u['UNK']=0
for key in train_b_u.keys():
    if (train_b_u[key]<= th):
        train_b_u['UNK']+=train_b_u[key]
        del train_b_u[key]  
        
voc_b=len(train_b_u)


train_bg_u['UNK']=0
for key in train_bg_u.keys():
    if (train_bg_u[key]<= th):
        train_bg_u['UNK']+=train_bg_u[key]
        del train_bg_u[key]  
        
voc_bg=len(train_bg_u)

#print voc_g,voc_b,voc_bg


# In[19]:

#modify training,test corpus

train_gn=[]
for sents in train_g:
    new =[ x if x in train_g_u else 'UNK' for x in sents] 
    train_gn.append(new)
    
train_bn=[]
for sents in train_b:
    new =[ x if x in train_b_u else 'UNK' for x in sents] 
    train_bn.append(new)
    
train_bgn=[]
for sents in train_bg:
    new =[ x if x in train_bg_u else 'UNK' for x in sents] 
    train_bgn.append(new)
    
test_gn=[]
for sents in test_g:
    new =[ x if x in train_g_u else 'UNK' for x in sents] 
    test_gn.append(new)
    
test_bn=[]
for sents in test_b:
    new =[ x if x in train_b_u else 'UNK' for x in sents] 
    test_bn.append(new)
    
    


# In[21]:

#bigram with frequency
bigram_g={}
for sent in train_gn:
    sent_len=len(sent)
    for j in range(sent_len-1):
        bigram=(sent[j],sent[j+1])
        if bigram in bigram_g :
            bigram_g[bigram]+=1
        else:
            bigram_g[bigram]=1
#voc_g=len(bigram_g)
            
bigram_b={}
for sent in train_bn:
    sent_len=len(sent)
    for j in range(sent_len-1):
        bigram=(sent[j],sent[j+1])
        if bigram in bigram_b :
            bigram_b[bigram]+=1
        else:
            bigram_b[bigram]=1
            
#voc_b=len(bigram_b)

bigram_bg={}
for sent in train_bgn:
    sent_len=len(sent)
    for j in range(sent_len-1):
        bigram=(sent[j],sent[j+1])
        if bigram in bigram_bg :
            bigram_bg[bigram]+=1
        else:
            bigram_bg[bigram]=1
            
voc_bg=len(bigram_bg)

#print voc_g,voc_b ,voc_bg          


# In[22]:

#trigram with frequency
trigram_g={}
for sent in train_gn:
    sent_len=len(sent)
    for j in range(sent_len-2):
        trigram=(sent[j],sent[j+1],sent[j+2])
        if trigram in trigram_g :
            trigram_g[trigram]+=1
        else:
            trigram_g[trigram]=1
#voc_g=len(trigram_g)
            
trigram_b={}
for sent in train_bn:
    sent_len=len(sent)
    for j in range(sent_len-2):
        trigram=(sent[j],sent[j+1],sent[j+2])
        if trigram in trigram_b :
            trigram_b[trigram]+=1
        else:
            trigram_b[trigram]=1
#voc_b=len(trigram_b)

trigram_bg={}
for sent in train_bgn:
    sent_len=len(sent)
    for j in range(sent_len-2):
        trigram=(sent[j],sent[j+1],sent[j+2])
        if trigram in trigram_bg :
            trigram_bg[trigram]+=1
        else:
            trigram_bg[trigram]=1
voc_bg=len(trigram_bg)

#print voc_g,voc_b ,voc_bg  


# In[23]:

# unigram model
norm_g1=np.sum(train_g_u.values())
train_gp1={}
for key in train_g_u.keys():
    prob  = train_g_u[key]/np.float(norm_g1)
    train_gp1[key]=prob

norm_b1=np.sum(train_b_u.values())
train_bp1={}
for key in train_b_u.keys():
    prob  = train_b_u[key]/np.float(norm_b1)
    train_bp1[key]=prob
    
norm_bg1=np.sum(train_bg_u.values())
train_bgp1={}
for key in train_bg_u.keys():
    prob  = train_bg_u[key]/np.float(norm_bg1)
    train_bgp1[key]=prob
    

# In[26]:

# bigram model
 
k=0.01
    
norm_g2=len(train_g_u.values())
train_gp2={}
ntrain_gp2={}
for key in bigram_g.keys():
    prob  =(bigram_g[key]+k)/(train_g_u[key[0]]+k*norm_g2)
    train_gp2[key]=prob
for key in train_g_u.keys():
    prob  = k/(train_g_u[key]+k*norm_g2)
    ntrain_gp2[key]=prob



    
norm_b2=len(train_b_u.values())
train_bp2={}
ntrain_bp2={}
for key in bigram_b.keys():
    prob  =(bigram_b[key]+k)/(train_b_u[key[0]]+k*norm_b2)
    train_bp2[key]=prob
for key in train_b_u.keys():
    prob  = k/(train_b_u[key]+k*norm_b2)
    ntrain_bp2[key]=prob

    
norm_bg2=len(train_bg_u.values())
train_bgp2={}
ntrain_bgp2={}
for key in bigram_bg.keys():
    prob  =(bigram_bg[key]+k)/(train_bg_u[key[0]]+k*norm_bg2)
    train_bgp2[key]=prob
for key in train_bg_u.keys():
    prob  = k/(train_bg_u[key]+k*norm_bg2)
    ntrain_bgp2[key]=prob

# In[]
# trigram model
k=0.01
norm_g3=np.square(norm_g2)

train_gp3={}
ntrain_gp3={}
for key in trigram_g.keys():
    prob  = (trigram_g[key]+k)/(bigram_g[key[0:2]]+(k*norm_g3))
    train_gp3[key]=prob
for key in bigram_g.keys():
    prob  = k/(bigram_g[key]+(k*norm_g3))
    ntrain_gp3[key]=prob


norm_b3=np.square(norm_b2)
train_bp3={}
ntrain_bp3={}
for key in trigram_b.keys():
    prob  = (trigram_b[key]+k)/(bigram_b[key[0:2]]+(k*norm_b3))
    train_bp3[key]=prob
for key in bigram_b.keys():
    prob  = k/(bigram_b[key]+(k*norm_b3))
    ntrain_bp3[key]=prob  
    
norm_bg3=np.square(norm_bg2)
train_bgp3={}
ntrain_bgp3={}
for key in trigram_bg.keys():
    prob  = (trigram_bg[key]+k)/(bigram_bg[key[0:2]]+(k*norm_bg3))
    train_bgp3[key]=prob
for key in bigram_bg.keys():
    prob  = k/(bigram_bg[key]+(k*norm_bg3))
    ntrain_bgp3[key]=prob 
    
# In[]
# unigram perplexity on test data
def perplex_un(test_data,train_p):
    p_log = 0;
    n= 0;
    for sents in test_data:
        length = len(sents)
        for i in range(0,length):
             n += 1
             key = sents[i]
             if key in train_p:
                   p_log+= np.log(train_p[key])

    ppxt=np.exp(-p_log/n)
    return ppxt


brown_p=perplex_un(test_bn,train_bp1)
print "perplexity_add_k_brown", brown_p
gutten_p=perplex_un(test_gn,train_gp1)
print "perplexity_add_k_gutten",gutten_p
comb_p_b=perplex_un(test_bn,train_bgp1)
print "perplexity_add_k_b+g_brown",comb_p_b
comb_p_g=perplex_un(test_gn,train_bgp1)
print "perplexity_add_k_b+g_gutten",comb_p_g
# In[28]:

# perplexity

#bigram

def perplex_bi(test_data,bigarm_data,train_p,ntrain_p): 
    n = 0
    #m=2*len(test_data)
    p_log=0
    for sents in test_data:
        l = len(sents)
        
        for i in range(l-1):
            n += 1
            key = (sents[i],sents[i+1])
            if key in bigarm_data:
                p_log += np.log(train_p[key])
            else:
                key_1 = sents[i]
                p_log += np.log(ntrain_p[key_1])

    ppx = np.exp(-p_log/n)
    
  
    return ppx


# In[32]:


brown_p=perplex_bi(test_bn,bigram_b,train_bp2,ntrain_bp2)
print "perplexity_add_k_brown", brown_p
gutten_p=perplex_bi(test_gn,bigram_g,train_gp2,ntrain_gp2)
print "perplexity_add_k_gutten",gutten_p
comb_p_b=perplex_bi(test_bn,bigram_bg,train_bgp2,ntrain_bgp2)
print "perplexity_add_k_b+g_brown",comb_p_b
comb_p_g=perplex_bi(test_gn,bigram_bg,train_bgp2,ntrain_bgp2)
print "perplexity_add_k_b+g_gutten",comb_p_g







# In[]
# generating trigram sentence

# in one keyword it will take bigram probability
start_ws = ["he"]
#ws = input("give a key word:--")
#start_ws=[ws]
key_bi =[]
val_bi =[]
#give corpus to generate sentence
bigram_data=train_bp2
######################
for key in bigram_data.keys():
    if key[0]==start_ws[0]:
        key_bi.append(key[1])
        val_bi.append(bigram_data[key])
if not len(val_bi) ==0:
    amx_val = np.argmax(val_bi)
    start_ws.append(key_bi[amx_val])
else:
    print("word doesn't exit. selecting start word as the")
    start_ws = ["he","said"]
        
#start_ws = ["i","was"]
start_ws_old=start_ws;
gen_sent = []
gen_sent.append(start_ws[0])
gen_sent.append(start_ws[1])

#give corpus to generate sentence
trigram_data=train_bp3
######################
keys_temp1 =list(trigram_data.keys())
keys_temp=[]

for key in keys_temp1:
    if not ((key[0]=='UNK') | (key[1]=='UNK') | (key[2]=='UNK')):
        keys_temp.append(key)


val_temp = []
for key in keys_temp:
    val_temp.append(trigram_data[key])


print("trigram Generated sentence is:")
num_token =40
count_token = 0
while(True):
    key_sel = []
    val_sel =[]
    for i in range(0,len(keys_temp)):
        if (keys_temp[i][0] == start_ws[0] ) & (keys_temp[i][1] == start_ws[1] ):
            key_sel.append(i)
            val_sel.append(val_temp[i])
    if len(val_sel)==0:
        print("trigram pair for this bigram doesn't exist")
        break;
        
    ind_max = key_sel[np.argmax(val_sel)]
    pred_word = keys_temp[ind_max][2]
    gen_sent.append(pred_word)
    start_ws = [start_ws[1],pred_word]
    del keys_temp[ind_max]
    del val_temp[ind_max]
    count_token = count_token + 1
    if (pred_word=='</s>'):
        start_ws = start_ws_old
    if (count_token==num_token) :
        break


filtered_sent = []
for i in range(0,len(gen_sent)):
    if not ((gen_sent[i] =='</s>') or (gen_sent[i] =='<s>')):
        filtered_sent.append(gen_sent[i])

        
print(' '.join(filtered_sent))
            
#gu
#br
#g+b
