import string
def rmv_punc(l1):
    # function to remove punctuations from a word
    tokens=[]
    indices=[]
    for i,j in enumerate(l1):
        if j[0] in string.punctuation:
            tokens.append(j[1:])
            indices.append(i)
        elif j[len(j)-1] in string.punctuation:
            tokens.append(j[:len(j)-1])
            indices.append(i)
    return tokens,indices
def tokenize(text):
    tokens=[]
    temp = text.splitlines()
    temp2 = []
    for i in temp :
        temp2.append(i.split())
    for i in temp2:
        for j in i :
            if len(j)!=0:
                tokens.append(j.lower())
    while (len(rmv_punc(tokens)[0])!=0):
        indices = rmv_punc(tokens)
        j=0
        for i in indices[1] :
            tokens[i]= indices[0][j]
            j+=1
    return tokens

def sort(t_c):
    #function to sort dictionary by value
    temp = []
    m = 0 # initializing a max value
    for i,j in t_c.items(): 
        temp.append((j,i))
        if j>m:
            m=j
    res = {} # final dictionary with values in descending order
    while m>0:
        for i in temp :
            if i[0] == m :
                res[i[1]]=i[0]
        m-=1
    return res
def pipe(t_c):
        for i,j in t_c.items():
            yield (i,j)
class Text_Analyzer(object):
    
    def __init__(self, text):
        self.text = text
        self.token_count = {}

    def analyze(self):
        tokens = tokenize(text)
        for i in tokens :
            c=0
            for j in tokens:
                if j == i :
                    c+=1
                    self.token_count[i]=c
        return self.token_count
    
    def topN(self, N):
        tmp = sort(self.token_count)
        topN=[]
        tmp2 = pipe(tmp)
        try:
            while N>0:
                topN.append(next(tmp2))
                N-=1
            return topN
        except StopIteration :
            return "Maximum number of items exceeded!"
def bigram(text,N):
    tmp = tokenize(text)
    tmp2=[]
    for i in range(len(tmp)-1) :
            tmp2.append([tmp[i],tmp[i+1]])
    tmp3 = []
    for i in tmp2:
        if i not in tmp3:
            tmp3.append(i)
    tmp4 = []
    for i in tmp3:
        c=0
        for j in tmp2:
            if i==j :
                c+=1
        tmp4.append((i,c))
    m=[]
    result = []
    for i in tmp4 :
        m.append(i[1])
    m.sort(reverse=True)
    m=list(set(m))
    m.sort(reverse=True)
    for i in m :
        for j in tmp4:
            if j[1]==i :
                result.append(j)
    result = result[:N]
    return result 

if __name__ == "__main__": 
    
    # Test Question 1
    text=''' There was nothing so VERY remarkable in that; nor did Alice
    think it so VERY much out of the way to hear the Rabbit say to
    itself, `Oh dear!  Oh dear!  I shall be late!'  (when she thought
    it over afterwards, it occurred to her that she ought to have
    wondered at this, but at the time it all seemed quite natural);
    but when the Rabbit actually TOOK A WATCH OUT OF ITS WAISTCOAT-
    POCKET, and looked at it, and then hurried on, Alice started to
    her feet, for it flashed across her mind that she had never
    before seen a rabbit with either a waistcoat-pocket, or a watch to
    take out of it, and burning with curiosity, she ran across the
    field after it, and fortunately was just in time to see it pop
    down a large rabbit-hole under the hedge.
    '''   
    print("Test Q1:")
    print(tokenize(text))
    print("\n\n")

    # Test Question 2
    analyzer=Text_Analyzer(text)
    print("Test Q2:")
    print(analyzer.analyze())
    print("\n")
    print(analyzer.topN(142))
    print("\n\n")

    #3 Test Question 3
    print("Test Q3:")
    top_bigrams=bigram(text,11)
    print(top_bigrams)
        