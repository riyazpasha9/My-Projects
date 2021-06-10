#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------
Application for scrapping information about:
- workspaces,
- models,
- users,

It uses Anaplan API v2, using basic authorization to get token for v2 authorization.
Model sizes and User_Access_Export applies to entities accessible by the user.
Thus user should have access to all workspaces and active models.
"""
import csv
import datetime
import json
import os
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
import shutil
import sys
import time

sys.path.append("C:\\Users\\pashar1\\Desktop\\Test folder\\New Anaplan_API Scripts")

import no_proxyauth_Anaplan_Connector as connector

CMH_MODELS = [
{"Name":"ACoE WoW", "WorkspaceID":"" , "ModelID":""}
] # data of CMH models, where final data is imported and processed
API_URL = r'https://api.anaplan.com/2/0/' # prefix to all requests to Anaplan V2 API
ARCHIVE_PATH = r'C:\Users\pashar1\Desktop\CMH\Central Management  Hub - Scrap Sizes\Archive' # path to primary archive, where files are generated and zipped
GECKO_PATH = r'C:\Users\pashar1\Desktop\CMH\Central Management  Hub - Scrap Sizes\geckodriver.exe' # path to must-have .exe file needed for Python-Firefox communication
CATEGORY_ID = r'' # Business Function ID in Anaplan Tenant

today = datetime.datetime.now().date().isoformat().replace("-", "_") # in YYYY_MM_DD
getHeaders = None

def convert_size(size):
    """Translate size given in kB and MB to GB"""
    
    if "kB" in size:
        return float(size.strip(" kB")) / (1024 * 1024)
    elif "MB" in size:
        return float(size.strip(" MB")) / (1024)
    else:
        return size.strip(" GB")

def login_browser():
    """Run browser and login to Anaplan"""
    
    browser = webdriver.Firefox(executable_path = GECKO_PATH, service_log_path = os.devnull, firefox_binary = FirefoxBinary(r'C:\Program Files (x86)\Mozilla Firefox\firefox.exe'))
    browser.get(r"https://sdp.anaplan.com/frontdoor/saml/avivasaml")
    
    time.sleep(60)
    
    return browser

def scrap_sizes(browser, Workspaces, Models):
    """Gets info about models sizes and workspace size used from Anaplan GUI"""

    for ws in Workspaces:
        browser.get(r"https://us1a.app.anaplan.com/core3007/anaplan/framework.jsp?selectedWorkspaceId=" + ws["id"] + "&takeAction=modelManagement")
        time.sleep(7)
        WebDriverWait(browser, 50).until(expected_conditions.presence_of_element_located((By.CLASS_NAME, "workspaceSummary")))
        workspace_message = browser.find_element_by_class_name("workspaceSummary").text
        ws["sizeUsed"] = convert_size(workspace_message[workspace_message.find(". ")+3 : workspace_message.find("of")-1])
        models_size = browser.find_element_by_id("dijit__CssStateMixin_0").text.splitlines()
        i = -1
        size = ""
        models_info = []
        while i < len(models_size)-1:
            i += 1
            if "OFFLINE" == models_size[i]:
                continue
            if "" == size:
                size = convert_size(models_size[i])
                continue
            models_info.append({"name" : models_size[i], "size" : size})
            size = ""
        for md in models_info:
            for model in Models:
                if model["currentWorkspaceId"] == ws["id"] and model["name"] == md["name"]:
                    model["size"] = md["size"]
                    break
    return (Workspaces, Models)

def get_objects(objects_name):
    """Requests for info about argument"""
    
    print('\nGetting {}...'.format(objects_name))
    url = API_URL + objects_name

    request = connector.get_anaplan_json(url, getHeaders)
    if not request.ok:
        print('{} skipped - error {})'.format(objects_name, request))
        return
    Objects = json.loads(request.text.encode('utf-8'))[objects_name]

    return Objects

def write_workspaces(Workspaces):
    """Saves info about all available workspaces to csv file"""

    print("Scanned workspaces: {}".format(len(Workspaces)))
    with open('{}\\adi_workspaces_{}.csv'.format(ARCHIVE_PATH, today), 'w', encoding='utf-8', newline='') as file:
        wr = csv.writer(file, dialect='excel', quoting=csv.QUOTE_ALL)

        wr.writerow(["Workspace ID", "Name", "Active", "Size", "Size used"])
        for ws in Workspaces:
            wr.writerow([ws["id"], ws["name"], ws["active"], ws["sizeAllowance"], ws["sizeUsed"] if "sizeUsed" in ws else "0"])
    print("File saved")

def write_models(Models):
    """Saves info about all available models to csv file"""

    print("Scanned models: {}".format(len(Models)))
    with open('{}\\adi_models_{}.csv'.format(ARCHIVE_PATH, today), 'w', encoding='utf-8', newline='') as file:
        wr = csv.writer(file, dialect='excel', quoting=csv.QUOTE_ALL)

        wr.writerow(["Model ID", "Name", "Workspace", "Master Model", "Status", "Size"])
        for model in Models:
            category = ''
            for cat in model["categoryValues"]:
                if cat["categoryId"] == CATEGORY_ID:
                    category = cat["id"]
                    break
            wr.writerow([model["id"], model["name"], model["currentWorkspaceId"], category, model["activeState"], model["size"] if "size" in model else "0"])
    print("File saved")
    
def upload_files(FilesInfo):
    """Uploads files to Anaplan CMH models"""

    print("\nUploading files to CMH...")
    for fi in FilesInfo:
        for model in CMH_MODELS:
            ret = connector.upload_file(model["WorkspaceID"], model["ModelID"], getHeaders, fi["id"], '{}\\{}_{}.csv'.format(ARCHIVE_PATH, fi["name"], today))
            if (ret is None) or (not ret.ok):
                print("Upload FAILED: {}".format(fi["name"]))
                
def run_actions(ActionsInfo):
    """Runs actions in Anaplan CMH Models"""

    print("\nRunning actions in CMH...\n")
    mapping_stream = connector.mapping_table_to_stream([["D4_Day.S ADI", today]])
    for ai in ActionsInfo:
        print("Running {}".format(ai["name"]))
        for model in CMH_MODELS:
            ret = connector.run_anaplan_action(model["WorkspaceID"], model["ModelID"], getHeaders, ai["id"], mapping_stream)
            if (ret is None) or (not ret.ok):
                print("Action FAILED: {}".format(ai["name"]))

def main():

    global getHeaders
    getHeaders = connector.login_token()
    #threading.Timer(REFRESH_TIME, refresh_header).start() #Commented, because now script runs under 30 mins, so no need to refresh header
    
    #---Getting basic info---

    Workspaces = get_objects('workspaces')
    Models = get_objects('models')

    #---Browser part--- # ugly, switch to API once achievable

    browser = login_browser()
    (Workspaces, Models) = scrap_sizes(browser, Workspaces, Models)
    os.system('tskill plugin-container')
    browser.quit()

    #---Save Progress---

    write_workspaces(Workspaces)
    write_models(Models)

    #---Imports and actions---

    upload_files(FilesInfo)
    run_actions(ActionsInfo)

    #---Logout and finish
    
    connector.logout_token(getHeaders)      

if __name__ == "__main__":
    
    FilesInfo = [{
            "id" : "113000000154",
            "name" : "adi_workspaces"
        }, {
            "id" : "113000000157",
            "name" : "adi_models"
        }]

    ActionsInfo = [{
            "id" : "112000000199",
            "name" : "",
            "type" : "imports"
        }, {
            "id" : "112000000196",
            "name" : "",
            "type" : "processes"
        }]
    main()
