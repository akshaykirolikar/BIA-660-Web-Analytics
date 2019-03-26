from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions 
import re
import matplotlib.pyplot as plt

executable_path = './driver/geckodriver'
driver = webdriver.Firefox(executable_path=executable_path)

def getData(movie_id): # scraping first page for movie reviews
    movie_id = "https://www.rottentomatoes.com/m/" + movie_id + "/reviews"
    data = []
    driver.get(movie_id)
    reviewData = driver.find_elements_by_class_name("col-xs-16.review_container")
    r = re.compile(r'\d+.+')
    for i in reviewData:
        a = i.text.split("\n")
        tmp = a[-1]
        a = a[:-1]
        tmp = r.findall(tmp)
        if len(tmp)==0:
            a.append(None)
        else:
            a.append(tmp[0])
        data.append(tuple(a))
    return data

def plot_data(data): #plotting data on year basis
    year_rating = {}
    for i in data:
        tmp = i[0].split(' ')
        tmp = tmp[-1]
        rating = i[-1]
        if type(rating)==str:
            try:
                rating = eval(rating)
                if rating>1:
                    rating=rating/10
                if tmp not in year_rating:
                    year_rating[tmp] = [rating]
                else:
                    year_rating[tmp].append(rating)
            except:
                pass
    for i in year_rating:
        year_rating[i]=sum(year_rating[i])/len(year_rating[i])
    x = list(year_rating.keys())
    x.sort()
    y = list(year_rating.values())
    y = y[::-1]
    plt.bar(x,y,color = 'teal')
    plt.title("Average rating by year")
    plt.xlabel("Years")
    plt.xticks(rotation = 45)
    plt.show()
    return year_rating


def getFullData(movie_id):
    data = []
    movie_id = "https://www.rottentomatoes.com/m/" + movie_id + "/reviews"
    while True: #dynamically crawling till last page for movie reviews
        try:    
            driver.get(movie_id)
            reviewData = driver.find_elements_by_class_name("col-xs-16.review_container")
            r = re.compile(r'\d+.+')
            for i in reviewData:
                a = i.text.split("\n")
                tmp = a[-1]
                a = a[:-1]
                tmp = r.findall(tmp)
                if len(tmp)==0:
                    a.append(None)
                else:
                    a.append(tmp[0])
                data.append(tuple(a))
            forward = driver.find_element_by_xpath('/html/body/div[6]/div[4]/div[2]/section/div/div/div[2]/div[1]/a[2]')
            forward.click()
            driver.implicitly_wait(30)
            movie_id = driver.current_url
        except:
            break
    return data

if __name__ == "__main__":
# Test Q1
    print("Q1 :")
    data=getData("finding_dory")
    print(data)
    print("\n\n")
# Test Q2
    print("Q2 :")
    plot_data(data)
    print("\n\n")
# Test Q3
    print("Q3 :")
    data=getFullData("harry_potter_and_the_half_blood_prince")
    print(len(data), data[-1])
    plot_data(data)

    driver.quit()

