from selenium import webdriver

# init the webdriver
driver = webdriver.Chrome('path_to_chromedriver')



driver.get('https://www.youtube.com/')

search_bar = driver.find_element_by_name('search_query')
search_bar.send_keys('thinspo subliminal')
search_bar.submit()

video_titles = driver.find_elements_by_xpath('//a[@id="video-title"]')
for title in video_titles:
    print(title.text)


# close webdriver
driver.quit()
