from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

# # Start Chrome
# driver = webdriver.Chrome()
# driver.get("https://www.google.com/search?q=google+snake")
#
# complete_CAPTCHA = input("Confirm CAPTCHA is complete (y): ")
# while(complete_CAPTCHA != "y"):
#     continue
#
#
# settings = input("Confirm settings have been configured (y): ")
# while(settings != "y"):
#     continue
#
# info = driver.execute_script("""
#     return {
#         title: document.title,
#         url: location.href,
#         canvases: [...document.querySelectorAll("canvas")].map(c => ({
#             width: c.width,
#             height: c.height
#         })),
#         elements: document.querySelectorAll("*").length
#     };
# """)
#
# print(info)
#
# time.sleep(5)
# pyautogui.press('up')
# pyautogui.press('down')
# time.sleep(2)
# driver.quit()