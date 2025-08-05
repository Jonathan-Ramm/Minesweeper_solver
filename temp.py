from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
import time

def test_headless_firefox():
    options = Options()
    options.add_argument("--headless")  # Wichtig
    driver = webdriver.Firefox(options=options)

    driver.get("https://example.com")
    time.sleep(2)
    print(driver.title)
    driver.quit()

if __name__ == "__main__":
    test_headless_firefox()
