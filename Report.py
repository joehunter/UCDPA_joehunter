

import webbrowser
import os

# open in a new tab, if possible
new = 2
url = "file://c:/UCDPA_joehunter/Output/report.htm"
#webbrowser.open(url, new=new)

webbrowser.get("C:/Program Files/Google/Chrome/Application/chrome.exe %s").open(url)