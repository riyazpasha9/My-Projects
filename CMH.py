#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

Application for scrapping information about:
- workspaces,
- models,
- users,

It uses Anaplan API v2, using basic authorization to get token for v2 authorization.
Model sizes and User_Access_Export applies to entities accessible by the user.
Thus user should have access to all workspaces and active models.
"""
import concurrent.futures
import csv
import datetime
import json
import os
import re
"""from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary"""
import shutil
import sys
import threading
import time
import traceback
import win32com.client
import zipfile
import zlib
import win32com.client as win32


sys.path.append("C:\\Users\\pashar1\\Desktop\\Test folder\\New Anaplan_API Scripts")


import no_proxyauth_Anaplan_Connector as connector

CMH_MODELS = [
#{"Name":"TEST ACoE WoW", "WorkspaceID":"" , "ModelID":""},
{"Name":"", "WorkspaceID":"" , "ModelID":""}
] # data of CMH models, where final data is imported and processed
ASIA_MODELS = [
{"ModelID":"" , "FileName":"reporting"}
,{"ModelID":"" , "FileName":"non_financial"}
] # data needed for special actions per model
API_URL = r'https://api.anaplan.com/2/0/'# prefix to all requests to Anaplan V2 API
ARCHIVE_PATH = r'C:\Users\pashar1\Desktop\CMH\Central Management Hub - Automated Daily Imports\Archive' # path to primary archive, where files are generated and zipped
TEMP_PATH = r'C:\Users\pashar1\Desktop\CMH\Central Management Hub - Automated Daily Imports\Temp' # path to temporary folder, it can be cleared if there are some garbage after run
REMOTE_ARCHIVE_PATH = r'C:\Users\pashar1\Desktop\CMH\Central Management Hub - Automated Daily Imports\Remote Archive' # path to remote drive where final backup is done
#GECKO_PATH = r'C:\Anaplan_API\API_Files\geckodriver.exe' # path to must-have .exe file needed for Python-Firefox communication
CUSTOMER_ID = r'' # Aviva Tenant ID in Anaplan, needed in case user running the script has access to tenant outside Aviva
CATEGORY_ID = r'' # Business Function ID in Anaplan Tenant
#REFRESH_TIME = 1740 # in sec, =29 min
CONNECTIONS = 32 # max concurent requests, Anaplan asked us to keep it low to avoid DoSing their infrastructure

today = datetime.datetime.now().date().isoformat().replace("-", "_") # in YYYY_MM_DD
mail_body = " "
getHeaders = None
log = None

def refresh_log():
    """Refreshes log file"""
    
    global log
    log.close()
    log = open('{}\\adi_log_{}.log'.format(ARCHIVE_PATH, today), 'a')
    sys.stdout = log

def report(text):
    """Print to log and to mail"""
    
    print(text)
    global mail_body
    mail_body += f"<br>{text}"

def err_handle(error):
    """Generic error handle"""
    
    print("\nERROR:\n{}".format(traceback.format_exc()))
    print("End: {}\n==================================".format(datetime.datetime.now()))
    global log
    log.close()
    os._exit(0)

def convert_size(size):
    """Translate size given in kB and MB to GB"""
    
    if "kB" in size:
        return float(size.strip(" kB")) / (1024 * 1024)
    elif "MB" in size:
        return float(size.strip(" MB")) / (1024)
    else:
        return size.strip(" GB")
"""
def login_browser():
    #Run browser and login to Anaplan

    creds = connector.decrypt_credentials((connector.read_credentials(connector.get_cert_path())))

    browser = webdriver.Firefox(executable_path = GECKO_PATH, service_log_path = os.devnull, firefox_binary = FirefoxBinary(r'C:\Program Files (x86)\Mozilla Firefox\firefox.exe'))
    browser.minimize_window()
    browser.get(r"https://us1a.app.anaplan.com/frontdoor/login?service=https%3A%2F%2Fsdp.anaplan.com%2Flaunchpad%2Fnonpublic%2Ftiles%2FCustomerSelect.action%3FcustomerGUID%3Da" + CUSTOMER_ID)
    time.sleep(15)
    username = browser.find_element_by_id('username')
    username.clear()
    username.send_keys(creds[0])
    password = browser.find_element_by_id('password')
    password.clear()
    password.send_keys(creds[1])
    time.sleep(1)
    password.send_keys(Keys.RETURN)
    time.sleep(10)

    return browser 

def scrap_sizes(browser, Workspaces, Models):
    #Gets info about models sizes and workspace size used from Anaplan GUI

    for ws in Workspaces:
        browser.get(r"https://us1a.app.anaplan.com/anaplan/framework.jsp?selectedWorkspaceId=" + ws["id"] + "&takeAction=modelManagement")
        time.sleep(15)
        WebDriverWait(browser, 15).until(expected_conditions.presence_of_element_located((By.CLASS_NAME, "workspaceSummary")))
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
    return (Workspaces, Models) """

def refresh_header():
    """Refreshes global token"""

    try:
        global getHeaders
        getHeaders = connector.refresh_token()
        #getHeaders = {'Authorization': connector.refresh_token(connector.make_session(getHeaders))}
        
        print("Token refreshed: {}".format(datetime.datetime.now()))
        refresh_log()
        threading.Timer(REFRESH_TIME, refresh_header).start()
    except Exception as error:
        err_handle(error)

def get_objects(objects_name):
    """Requests for info about argument"""
    
    print('\nGetting {}...'.format(objects_name))
    refresh_log()
    url = API_URL + objects_name

    request = connector.get_anaplan_json(url, getHeaders)
    if not request.ok:
        print('{} skipped - error {})'.format(objects_name, request))
        return
    Objects = json.loads(request.text.encode('utf-8'))[objects_name]

    return Objects

def get_me():
    """Gets info about current user from Anaplan API"""

    print("\nGetttng me...")
    refresh_log()
    url = API_URL + 'users/me'

    request = connector.get_anaplan_json(url, getHeaders)
    if not request.ok:
        print('Users/me skipped - error {})'.format(request))
        return
    myInfo = json.loads(request.text.encode('utf-8'))["user"]
    print("My email: " + myInfo["email"])

    return myInfo

def write_admin_info(myEmail, startTime, crawlEndTime):
    """Saves info about current run to csv file"""

    print("Saving admin_info...")
    refresh_log()
    with open('{}\\adi_admin_info_{}.csv'.format(ARCHIVE_PATH, today), 'w', encoding='utf-8', newline='') as file:
        wr = csv.writer(file, dialect='excel', quoting=csv.QUOTE_ALL)

        wr.writerow(["Line Items", "Values"])
        wr.writerow(["Admin", myEmail])
        wr.writerow(["Start Time", startTime])
        wr.writerow(["End Time", crawlEndTime])
    print("File saved")

def write_workspaces(Workspaces):
    """Saves info about all available workspaces to csv file"""

    report("Scanned workspaces: {}".format(len(Workspaces)))
    refresh_log()
    with open('{}\\adi_workspaces_{}.csv'.format(ARCHIVE_PATH, today), 'w', encoding='utf-8', newline='') as file:
        wr = csv.writer(file, dialect='excel', quoting=csv.QUOTE_ALL)

        wr.writerow(["Workspace ID", "Name", "Active", "Size", "Size used"])
        for ws in Workspaces:
            wr.writerow([ws["id"], ws["name"], ws["active"], ws["sizeAllowance"], ws["sizeUsed"] if "sizeUsed" in ws else "0"])
    print("File saved")

def write_models(Models):
    """Saves info about all available models to csv file"""

    report("Scanned models: {}".format(len(Models)))
    refresh_log()
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

def write_users(Users):
    """Saves info about all available users to csv file"""

    report("Scanned users: {}".format(len(Users)))
    refresh_log()
    with open('{}\\adi_users_{}.csv'.format(ARCHIVE_PATH, today), 'w', encoding='utf-8', newline='') as file:
        wr = csv.writer(file, dialect='excel', quoting=csv.QUOTE_ALL)

        wr.writerow(["Email", "First Name", "Last Name", "Active", "Last Login", "User ID"])
        for user in Users:
            wr.writerow([user["email"], user["firstName"], user["lastName"], user["active"], user["lastLoginDate"] if "lastLoginDate" in user else "", user["id"]])
    print("File saved")

def get_workspaces_access(Workspaces):
    """Scraps all available workspaces for info about users and admins"""

    print("\nScrapping workspaces_access...")
    refresh_log()
    WorkspacesAccess = []

    for ws in Workspaces:
        WorkspaceUsers = []
        WorkspaceAdmins = []

        url = API_URL + 'workspaces/{}/users'.format(ws["id"])
        request = connector.get_anaplan_json(url, getHeaders, False)
        if not request.ok:
            report('Workspace users skipped - error {}: {} (id: {})'.format(request,ws["name"], ws["id"]))
        else:
            WorkspaceOnlyUsers = json.loads(request.text.encode('utf-8'))["users"]
            for user in WorkspaceOnlyUsers:
                WorkspaceUsers.append({"email":user["email"], "access":"True", "admin":"False"})

        url = API_URL + 'workspaces/{}/admins?limit=2137'.format(ws["id"])
        request = connector.get_anaplan_json(url, getHeaders, False)
        if not request.ok:
            report('Workspace admins skipped - error {}: {} (id: {})'.format(request,ws["name"], ws["id"]))
            continue
        WorkspaceOnlyAdmins = json.loads(request.text.encode('utf-8'))["admins"]

        for admin in WorkspaceOnlyAdmins:
            is_user = False
            for access in WorkspaceUsers:
                if admin["email"] == access["email"]:
                    access["admin"] = "True"
                    is_user = True
                    break
            if not is_user:
                WorkspaceAdmins.append({"email":admin["email"], "access":"False", "admin":"True"})

        WorkspacesAccess.append({"id":ws["id"], "users" : (WorkspaceUsers + WorkspaceAdmins)})

    return WorkspacesAccess

def write_workspaces_access(WorkspacesAccess):
    """Saves info about workspaces access - users and admins - to csv file"""

    print("Saving workspaces_access...")
    refresh_log()
    with open('{}\\adi_workspaces_access_{}.csv'.format(ARCHIVE_PATH, today), 'w', encoding='utf-8', newline='') as file:
        wr = csv.writer(file, dialect='excel', quoting=csv.QUOTE_ALL)

        wr.writerow(["Workspace ID", "Email", "Has access", "Is admin"])
        for ws in WorkspacesAccess:
            for user in ws["users"]:
                wr.writerow([ws["id"], user["email"], user["access"], user["admin"]])
    print("File saved")

def get_users_model(model):
    """Extract user models"""
    return {"id":model["id"], "name":model["name"],"request":connector.get_anaplan_json(API_URL + 'models/{}/users'.format(model["id"]), getHeaders, False)}   

def get_users_models(Models):
    """Scraps all users for their access to models"""

    print("\nScrapping users_models...")
    refresh_log()
    ActiveModelsCount = 0
    futures_list =[]
    UsersModels = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as executor:
        for model in Models:
            if model["activeState"]!="ARCHIVED":
                ActiveModelsCount += 1
                futures_list.append(executor.submit(get_users_model, model))
        for future in concurrent.futures.as_completed(futures_list):
            res = future.result()
            if not res["request"].ok:
                report('Model skipped - error {} in: {} (id: {})'.format(res["request"], res["name"], res["id"]))
                continue
            UsersModels.append({"users":json.loads(res["request"].text.encode('utf-8'))["users"],"id":res["id"]})

    report('Number of active models: {}'.format(ActiveModelsCount))
    return UsersModels

def write_users_models(UsersModels):
    """Saves info about users access to models to csv file"""

    print("Saving users_models...")
    refresh_log()
    with open('{}\\adi_users_models_{}.csv'.format(ARCHIVE_PATH, today), 'w', encoding='utf-8', newline='') as file:
        wr = csv.writer(file, dialect='excel', quoting=csv.QUOTE_ALL)

        wr.writerow(["Email", "Model ID", "Has access"])
        for model in UsersModels:
            for user in model["users"]:
                wr.writerow([user["email"], model["id"], "True"])
    print("File saved")

def user_access_export(model):
    """User_Access_Export - run & download"""
    
    export_id = None
    model_file_path = '{}\\adi_UAE_{}_{}.csv'.format(TEMP_PATH, model["id"],today)
    
    request = connector.get_anaplan_json(API_URL + 'workspaces/{}/models/{}/exports'.format(model["currentWorkspaceId"],model["id"]), getHeaders, False)
    if not request.ok:
        report('UAE Model skipped -  error {}: {} (id: {})'.format(request, model["name"], model["id"]))
        return None
    Exports = json.loads(request.text.encode('utf-8'))
    if not Exports["meta"]["paging"]["currentPageSize"] == 0:
        Exports = Exports["exports"]
        for info in Exports:
            if info["name"] == "User_Access_Export":
                connector.run_anaplan_action(model["currentWorkspaceId"], model["id"], getHeaders, info["id"], None, False)
                if connector.download_file(model["currentWorkspaceId"], model["id"], getHeaders, info["id"], model_file_path, False):
                    return {"id" : model["id"], "model_file_path" : model_file_path}
                report('UAE Model skipped -  cannot download the file from: {} (id: {})'.format(model["name"], model["id"]))
                return None

    report('UAE Model skipped - no User_Access_Export action in: {} (id: {})'.format(model["name"], model["id"]))
    return None

def user_access_exports(userID):
    """Downloads all available User_Access_Exports"""

    print("\nScrapping User_Access_Exports...")

    myActiveModelsCount = 0
    url = API_URL + 'users/{}/models'.format(userID)
    UAEModels = []
    futures_list = []

    request = connector.get_anaplan_json(url, getHeaders)
    if not request.ok:
        print('User_Access_Exports skipped -  error {}'.format(request))
        return
    myModels = json.loads(request.text.encode('utf-8'))["models"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as executor:
        for model in myModels:
            if model["activeState"] != "ARCHIVED":
                myActiveModelsCount += 1
                futures_list.append(executor.submit(user_access_export, model))
        for future in concurrent.futures.as_completed(futures_list):
            res = future.result()
            if res is not None:
                UAEModels.append(res)

    report('Number of my models: {}'.format(len(myModels)))
    report('Number of my active models: {}'.format(myActiveModelsCount))
    report('Number of User_Access_Exports run: {}'.format(len(UAEModels)))
    print("Scrapping User_Access_Exports finished")

    return UAEModels

def write_asia_user_access_exports(UAEModels):
    """Copies relevant UAE to import them individually"""

    print("Copying Asia files...")
    for asia_model in ASIA_MODELS:
        check = 0
        for model in UAEModels:
            if asia_model["ModelID"] == model["id"]:
                shutil.copy2(model["model_file_path"], '{}\\{}_{}.csv'.format(ARCHIVE_PATH, asia_model["FileName"], today))
                check = 1
                break
        if 0 == check :
            report('Asia Model skipped -  cannot find the file for: {} (id: {})'.format(asia_model["FileName"], asia_model["ModelID"]))
        
    print("Files copied")

def write_user_access_exports(UAEModels):
    """Merges info from User_Access_Exports to one csv file"""
    
    print("Merging User_Access_Exports files...")
    refresh_log()
    with open('{}\\adi_user_access_export_{}.csv'.format(ARCHIVE_PATH, today), 'a', encoding='utf-8', newline='') as file:
        wr = csv.writer(file, dialect='excel', quoting=csv.QUOTE_ALL)
        wr.writerow(["Model ID", "Email", "Role", "Is admin", "Is SSO"])
        for model in UAEModels:
            with open(model["model_file_path"], 'r', encoding='utf-8') as source:
                next(source)
                rd = csv.reader(source,dialect='excel')
                for row in rd:
                    wr.writerow([model["id"],row[0],row[3],row[-3],row[-2]])
            os.remove(model["model_file_path"])
    print("File saved")

def write_new_day(status):
    """Saves to csv file info needed to add current day to subset"""

    print("Saving new_day={}...".format(status))
    refresh_log()
    with open('{}\\adi_new_day_{}.csv'.format(ARCHIVE_PATH, today), 'w', encoding='utf-8', newline='') as file:
        wr = csv.writer(file, dialect='excel', quoting=csv.QUOTE_ALL)

        wr.writerow(["T4_Day", "T4_Day.S ADI"])
        wr.writerow([today, status])
    print("File saved")

def upload_and_run_new_day_removal(file_info, action_info):
    """Upload and run action to remove today from subset - prevents data mess when running script more than once"""

    print("\nRemoving new_day from subset...")
    for model in CMH_MODELS:
        ret = connector.upload_file(model["WorkspaceID"], model["ModelID"], getHeaders, file_info["id"], '{}\\{}_{}.csv'.format(ARCHIVE_PATH, file_info["name"], today))
        if (ret is None) or (not ret.ok):
            report("Upload FAILED: {}".format(file_info["name"]))
        ret = connector.run_anaplan_action(model["WorkspaceID"], model["ModelID"], getHeaders, action_info["id"])
        if (ret is None) or (not ret.ok):
            report("Action FAILED: {}".format(action_info["name"]))

def upload_files(FilesInfo):
    """Uploads files to Anaplan CMH models"""

    print("\nUploading files to CMH...")
    refresh_log()
    for fi in FilesInfo:
        for model in CMH_MODELS:
            ret = connector.upload_file(model["WorkspaceID"], model["ModelID"], getHeaders, fi["id"], '{}\\{}_{}.csv'.format(ARCHIVE_PATH, fi["name"], today))
            if (ret is None) or (not ret.ok):
                report("Upload FAILED: {}".format(fi["name"]))
                
def run_actions(ActionsInfo):
    """Runs actions in Anaplan CMH Models"""

    print("\nRunning actions in CMH...\n")
    mapping_stream = connector.mapping_table_to_stream([["D4_Day.S ADI", today]])
    refresh_log()
    for ai in ActionsInfo:
        print("Running {}".format(ai["name"]))
        for model in CMH_MODELS:
            ret = connector.run_anaplan_action(model["WorkspaceID"], model["ModelID"], getHeaders, ai["id"], mapping_stream)
            if (ret is None) or (not ret.ok):
                report("Action FAILED: {}".format(ai["name"]))

def find_file_paths():
    """Lists all files with matching names in ARCHIVE_PATH"""
    
    file_paths = []
    filenames = os.listdir(ARCHIVE_PATH)
    for filename in filenames:
        if re.search("^adi_.*_" + today + "\.(csv|log)$", filename):
            file_paths.append(os.path.join(ARCHIVE_PATH, filename))

    return file_paths

def zip_and_archive(file_paths):
    """Zips given files to archive, add there old archive if it's possible"""
    
    zip_path = '{}\\CMH_ADI_{}'.format(ARCHIVE_PATH, today)
        
    if os.path.isfile(zip_path + '.zip'):
        os.rename(zip_path + '.zip', zip_path + '_OLD.zip')
        file_paths.append(zip_path + '_OLD.zip')

    zip_path += '.zip'

    zlib.Z_DEFAULT_COMPRESSION = zlib.Z_BEST_COMPRESSION
        
    with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zip: 
        for file in file_paths:
            zip.write(file, os.path.relpath(file, ARCHIVE_PATH))

    #delete files
    for file in file_paths:
        os.remove(file)

    #copy zip
    if os.path.isdir(REMOTE_ARCHIVE_PATH):
        try:
            shutil.copy2(zip_path, REMOTE_ARCHIVE_PATH)
            return zip_path, '{}\\CMH_ADI_{}.zip'.format(REMOTE_ARCHIVE_PATH, today)
        except:
            pass
    report('Cannot copy zip to remote location {} - {} has not been copied!'.format(REMOTE_ARCHIVE_PATH, zip_path))
    return (zip_path, None)

def send_mail(zip_paths):
    """Sends email with run summary"""
    
    Outlook = win32.Dispatch('outlook.application')    
    mail = Outlook.CreateItem(0)
    mail.To = ""
    mail.Subject = "CMH ADI Script: {}".format(today)
    mail.HTMLBody = "Hello ACoE,<br><br>As you can see, I got the job done. You can find output files in archive on <a href='{}'>remote PC</a>, ".format(zip_paths[0])
    mail.HTMLBody += ("or on " if zip_paths[1] else "but today there was a problem with making backup to ")
    mail.HTMLBody += "<a href='{}'>shared G drive</a>.<br>".format((zip_paths[1] if zip_paths[1] else REMOTE_ARCHIVE_PATH))
    mail.HTMLBody += "You can access the target CMH model via link:"
    for model in CMH_MODELS:
        mail.HTMLBody += "<a href='app.anaplan.com/a/springboard-platform-gateway-service/workspaces/{}/models/{}/redirect'> {}</a>".format(model["WorkspaceID"],model["ModelID"],model["Name"])
    mail.HTMLBody += mail_body
    mail.HTMLBody += "<br><br>Regards,<br>Python<br>Central Management Hub - Automated Daily Imports"
    mail.Send()
    #time.sleep(15)
    #os.system(r'"C:\Program Files (x86)\Microsoft Office\Office16\OUTLOOK.EXE"')
    #time.sleep(60)
    #os.system("Taskkill /IM OUTLOOK.EXE")

def main():

    try:
        #---Prepare and login---
        stdbuff = sys.stdout
        global log
        log = open('{}\\adi_log_{}.log'.format(ARCHIVE_PATH, today), 'a')
        sys.stdout = log
        
        startTime = datetime.datetime.now()
        print("==================================\nStart: {}".format(startTime))

        global getHeaders
        getHeaders = connector.login_token()
        #threading.Timer(REFRESH_TIME, refresh_header).start() #Commented, because now script runs under 30 mins, so no need to refresh header
        
        #---Getting basic info---
        
        myInfo = get_me()
        Workspaces = get_objects('workspaces')
        Models = get_objects('models')
        Users = get_objects('users')

        #---Browser part--- # ugly, switch to API once achievable

##        browser = login_browser()
##        (Workspaces, Models) = scrap_sizes(browser, Workspaces, Models)
##        os.system('tskill plugin-container')
##        browser.quit()

        #---Save Progress---

        write_workspaces(Workspaces)
        write_models(Models)
        write_users(Users)

        #---Scrapping part---
        WorkspacesAccess = get_workspaces_access(Workspaces)
        write_workspaces_access(WorkspacesAccess)

        UsersModels = get_users_models(Models)
        write_users_models(UsersModels)

        UAEModels = user_access_exports(myInfo["id"])
        write_asia_user_access_exports(UAEModels)
        write_user_access_exports(UAEModels)

        crawlEndTime = datetime.datetime.now()
        print("\nCrawling finished: {}".format(crawlEndTime))

        write_admin_info(myInfo["email"], startTime, crawlEndTime)

        #---Imports and actions---
        write_new_day("False")
        upload_and_run_new_day_removal(FilesInfo[0], ActionsInfo[0])
        write_new_day("True")
        upload_files(FilesInfo)
        run_actions(ActionsInfo)

        #---Logout and finish
        
        connector.logout_token(getHeaders)

        print("End: {}\n==================================".format(datetime.datetime.now()))
        log.close()
        sys.stdout = stdbuff
                
        zip_paths = zip_and_archive(find_file_paths())
        send_mail(zip_paths)        
        os._exit(0)

    except Exception as error:
        err_handle(error)        

if __name__ == "__main__":
    
    FilesInfo = [{
            "id" : "",
            "name" : "adi_new_day"
        }, {
            "id" : "",
            "name" : "adi_admin_info"
        }, {
            "id" : "",
            "name" : "adi_users"
        }, {
            "id" : "113000000166",
            "name" : "adi_user_access_export"
        }, {
            "id" : "",
            "name" : "adi_workspaces"
        }, {
            "id" : "",
            "name" : "adi_workspaces_access"
        }, {
            "id" : "",
            "name" : "adi_models"
        }, {
            "id" : "",
            "name" : "adi_users_models"
        }, {
            "id" : "",
            "name" : "adi_asia_reporting"
        }, {
            "id" : "",
            "name" : "adi_asia_non_financial"
        }]

    ActionsInfo = [{
            "id" : "",
            "name" : "ADI.0 D4_Day.S ADI",
            "type" : "imports"
        }, {
            "id" : "",
            "name" : "1. CMH  - Automated Daily Import",
            "type" : "processes"
        }]
    main()
