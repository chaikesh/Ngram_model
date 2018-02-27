# coding: utf-8

# In[1]:

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






# In[]

#specify corpus
unigram_d=train_bg_u
bigram_d =bigram_bg

 # unigram continuation probability
 
pc_n= {}
pc_d = {}
pc = {}

for key2 in bigram_d.keys():
    if key2[1] in pc_n.keys():
        pc_n[key2[1]] =pc_n[key2[1]] + 1
        pc_d[key2[1]] =pc_d[key2[1]] + unigram_d[key2[0]]
    else:
        pc_n[key2[1]] = 1
        pc_d[key2[1]] = unigram_d[key2[0]]

for key in pc_n.keys():
    pc[key] =   pc_n[key]/np.float(pc_d[key])
    



# unigram lamda
d = 0.75
l_n= {}
for key2 in bigram_d.keys():
    if key2[0] in l_n:
        l_n[key2[0]] = l_n[key2[0]] + 1
    else:
        l_n[key2[0]] =  1

l={}
for key in l_n.keys():
    l[key] = d*l_n[key]/unigram_d[key]

# probability calculation
d = 0.75
pkn= {}
for key2 in bigram_d.keys():
    pkn[key2] = ((bigram_d[key2] -d)/unigram_d[key2[0]]) + l[key2[0]]*pc[key2[1]]
 
# In[]

  #  perplexity 
  

#p_log = 0;
#n = 0;
#for sents in test_bn:
    #length = len(sents)
    #for i in range(0,length-1):
       # n+=1
       # key = (sents[i],sents[i+1])
       # if key in bigram_d.keys():
            
            #p_log += np.log(pkn[key])
       # else:
            #prob = l[key[0]]*pc[key[1]]
           # p_log += np.log(prob)

#ppx=np.exp(-p_log/n)
#print ppx


# In[ ]:# generating trigram sentence



#give corpus to generate sentence
bigram_data=pkn
######################

start_ws = ["he"]
key_bi =[]
val_bi =[]
gen_sent = []
gen_sent.append(start_ws[0])


#######
keys_temp1 =list( bigram_data.keys())
keys_temp=[]

for key in keys_temp1:
    if not ((key[0]=='UNK') | (key[1]=='UNK') ):
        keys_temp.append(key)

val_temp = []
for key in keys_temp:
    val_temp.append(bigram_data[key])


print("bigram Generated sentence is:")
num_token =50
count_token = 0
while(True):
    key_sel = []
    val_sel =[]
    for i in range(0,len(keys_temp)):
        if (keys_temp[i][0] == start_ws[0] ) :
            key_sel.append(i)
            val_sel.append(val_temp[i])
    if len(val_sel)==0:
        print("bigram pair for this bigram doesn't exist")
        break;
    ind_max = key_sel[np.argmax(val_sel)]
    pred_word = keys_temp[ind_max][1]
    gen_sent.append(pred_word)
    start_ws = [pred_word]
    del keys_temp[ind_max]
    del val_temp[ind_max]
    count_token = count_token + 1
    if (pred_word=='</s>'):
        start_ws = "he"
    if (count_token==num_token) :
        break


filtered_sent = []
for i in range(0,len(gen_sent)):
    if not ((gen_sent[i] =='</s>') or (gen_sent[i] =='<s>')):
        filtered_sent.append(gen_sent[i])

        
print(' '.join(filtered_sent))
            




