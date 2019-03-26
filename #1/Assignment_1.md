
# <center>Assignment 1</center>

## Q1. Define a function to analyze the frequency of words in a string ##
 - Define a function named "**tokenize**" which does the following:
     * has a string as an input 
     * splits the string into a list of tokens by space. For example, "it's hello world!" will be split into two tokens ["it's", "hello","world!"]
     * removes all spaces around each token (including tabs, newline characters ("\n"))
     * if a token starts with or ends with a punctuation, remove the punctuation, e.g. "world<font color="red">!</font>" -> "world",  "<font color="red">'</font>hello<font color="red">'</font>"->"hello" (<font color="blue">hint, you can use *string.punctuation* to get a list of punctuations, where *string* is a module you can import</font>)
     * removes empty tokens, i.e. *len*(token)==0
     * converts all tokens into lower case
     * returns all the tokens as a list output
    

## Q2. Define a class to analyze a document ##
 - Define a new class called "**Text_Analyzer**" which does the following :
    - has two attributes: 
        * **input_string**, which receives the string value passed by users when creating an object of this class.
        * **token_count**, which is set to {} when an object of this class is created.
        
    - a function named "**analyze**" that does the following:
      * calls the function "tokenize" to get a list of tokens. 
      * creates a dictionary containing the count of every unique token, e.g. {'it': 5, 'hello':1,...}
      * saves this dictionary to the token_count attribute
    - a function named "**topN**" that returns the top N words by frequency
      * has a integer parameter *N*  
      * returns the top *N* words and their counts as a list of tuples, e.g. [("hello", 5), ("world", 4),...] (<font color="blue">hint: By default, a dictionary is sorted by key. However, you need to sort the token_count dictionary by value</font>)
  
- What kind of words usually have high frequency? Write your analysis.      

## Q3. (Bonus) Create Bigrams from a document ##

A bigram is any pair of consecutive tokens in a document. Phrases are usually bigrams. Let's design a function to find phrases.
 - Create a new function called "**bigram**" which does the following :
     * takes a **string** and an integer **N** as inputs
     * calls the function "tokenize" to get a list of tokens for the input string
     * slice the list to get any two consecutive tokens as a bigram. For example ["it's", "hello","world"] will generate two bigrams: [["it's", "hello"],["hello","world"]]
     * count the frequency of each unique bigram
     * return top N bigrams and their counts 
 - Are you able to find good phrases from the top N bigrams? Write down your analysis in a document.

## Submission Guideline##
- Following the solution template provided below. Use __main__ block to test your functions and class
- Save your code into a python file (e.g. assign1.py) that can be run in a python 3 environment. In Jupyter Notebook, you can export notebook as .py file in menu "File->Download as".
- Make sure you have all import statements. To test your code, open a command window in your current python working folder, type "python assign1.py" to see if it can run successfully.
- For more details, check assignment submission guideline on Canvas
