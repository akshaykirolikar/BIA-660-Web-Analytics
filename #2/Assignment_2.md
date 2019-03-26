
# <center>Assignment 2</center>

## Q1. Define a function to analyze a numpy array
 - Assume we have an array (with shape (M,N)) which contains term frequency of each document, where each row is a document, each column is a word, and the corresponding value denotes the frequency of the word in the document. Define a function named "analyze_tf_idf" which:
      * takes the **array**, and an integer **K** as the parameters.
      * normalizes the frequency of each word as: word frequency divided by the length of the document. Save the result as an array named **tf** (i.e. term frequency)
      * calculates the document frequency (**df**) of each word, e.g. how many documents contain a specific word
      * calculates **tf_idf** array as: **tf / (log(df)+1)** (tf divided by log(df)). The reason is, if a word appears in most documents, it does not have the discriminative power and often is called a "stop" word. The inverse of df can downgrade the weight of such words.
      * for each document, finds out the **indexes of words with top K largest values in the tf_idf array**, ($0<K<=N$). These indexes form an array, say **top_K**, with shape (M, K)
      * returns the tf_idf array, and the top_K array.
 - Note, for all the steps, ** do not use any loop**. Just use array functions and broadcasting for high performance computation.

## Q2. Define a function to analyze stackoverflow dataset using pandas
 - Define a function named "analyze_data" to do the follows:
   * Take a csv file path string as an input. Assume the csv file is in the format of the provided sample file (question.csv).
   * Read the csv file as a dataframe with the first row as column names
   * Find questions with top 3 viewcounts among those answered questions (i.e answercount>0). Print the title and viewcount columns of these questions.
   * Find the top 5 users (i.e. quest_name) who asked the most questions.
   * Create a new column called "first_tag" to store the very first tag in the "tags" column (hint: use "apply" function; tags are separted by ", ")
   * Show the mean, min, and max viewcount values for each of these tags: "python", "pandas" and "dataframe"
   * Create a cross tab with answercount as row indexes, first_tag as column names, and the count of samples as the value. For "python" question (i.e. first_tag="python"), how many questions were not answered (i.e., answercount=0), how many questions were answered once (i.e., answercount=1), and how many questions were anasered twice  (i.e., answercount=2)? Print these numbers.
 - This function does not have any return. Just print out the result of each calculation step.

## Q3 (Bonus). Analyzed a collection of documents
 - Define a function named "analyze_corpus" to do the follows:
   * Similar to Q2, take a csv file path string as an input. Assume the csv file is in the format of the provided sample file (question.csv).
   * Read the "title" column from the csv file and convert it to lower case
   * Split each string in the "title" column by space to get tokens. Create an array where each row represents a title, each column denotes a unique token, and each value denotes the count of the token in the document
   * Call your function in Q1 (i.e. analyze_tf_idf) to analyze this array
   * Print out the top 5 words by tf-idf score for the first 20 questions. Do you think these top words allow you to find similar questions or differentiate a question from dissimilar ones? Write your analysis as a pdf file.
   
- This function does not have any return. Just print out the result if asked.
   
