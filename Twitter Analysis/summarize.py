#!/usr/bin/env python
# coding: utf-8

# In[1]:


def main():
    collect = open('collect-summary.txt','r',encoding = 'utf-8')
    cluster = open('clusters.txt','r',encoding = 'utf-8')
    classify = open('classify.txt','r',encoding = 'utf-8')
    summary = open('summary.txt' , 'w', encoding = 'utf-8')
    summary.write(collect.read())
    summary.write('\n')
    summary.write(cluster.read())
    summary.write('\n')
    summary.write(classify.read())
    summary.write('\n')
    summary.close()
    collect.close()
    cluster.close()
    classify.close()

if __name__ == "__main__":
    main()
    

