"""

Riyaz Pasha

The code will Run the exports from the AIRE 2.0 Model. All the Qlik Exports will goto the Profolio Management Folder.

"""
import csv
import datetime
import json
import os
import shutil
import sys
import time
import glob
import re
import win32com.client as win32
import win32com.client
import pythoncom

#Path of the AnaplanAPIConnector_v2_0 script
sys.path.append("C:\\Users\\pashar1\\Desktop\\Test folder\\New Anaplan_API Scripts")

import no_proxyauth_Anaplan_Connector as connector

OUTPUT_PATH = "X:\\Extracts"
LOG_PATH = "users\\Export.log" #path to the log file
#MAIN_FOLDER_PATH = "C:\\Users\\pashar1\\Desktop\\Test folder\\New Anaplan_API Scripts\\Real Estate -  Exports\\Qilk Extracts Main Folder"
MODEL = {"WorkspaceID":"" , "ModelID":""}

mail_body = ''

OK = "SUCCESSFUL"
NOT_OK = "FAILED"
today = datetime.datetime.now().date().isoformat().replace("-", "_") # in YYYY_MM_DD

def get_timestamp():
    """Simple function to always get timestamp in the same format"""
    
    return datetime.datetime.now().isoformat()[:19] # in YYYY-MM-DDTHH:MM:SS

def log_text(text):
    """Helper to write line of text to the log"""
    
    with open(LOG_PATH, 'a') as log:
        log.write(f"[{get_timestamp()}] {text}\n")

def report(text):
    """Helper to report information to relevant streams"""

    log_text(text)
    print(text)
    
    global mail_body
    mail_body += text.replace(OK,
                              f"<span style=\"color:#00FF00;\">{OK}</span>").replace(NOT_OK,
                                                                                          f"<span style=\"color:#FF0000;\">{NOT_OK}</span>") + "<br>"
def remove_previous(OUTPUT_PATH):
    """delete previous files from the Main folder"""
    file_paths = []
    filenames = os.listdir(OUTPUT_PATH)
    for filename in filenames:
        if re.search("(^Qlik|^Export).*\.(csv|log)$", filename):
            file_paths.append(os.path.join(OUTPUT_PATH, filename))
    
    for file in file_paths:
        os.remove(file)

    report(f"All the previous extracts has been removed {OK} from : {OUTPUT_PATH}")
    report(f"All the previous Extract are being saved in the :{OUTPUT_PATH}\\Qlik Exports Historic Data")
    report(f"Below files are being downloaded to the path : {OUTPUT_PATH}")
    
    
def run_actions(action_info, file_info, user_token):

        

        for action in action_info:
            
            ret = connector.run_anaplan_action(MODEL["WorkspaceID"], MODEL["ModelID"], user_token, action['id'])
            if (ret is None) or (not ret.ok):
                report("Action FAILED: {}".format(action["name"]))

        for file in file_info:

            model_file_path = '{}\\{}.csv'.format(OUTPUT_PATH,file['name'])
            download = connector.download_file(MODEL["WorkspaceID"],  MODEL["ModelID"] ,user_token , file["id"],model_file_path)
            if download is True:
                report(f"Download {OK}: {file['name']}")

            else:
                report(f"Download {NOT_OK}: {file['name']}")
                       
        

def find_file_paths():
    """Lists all files with matching names in OUTPUT_PATH"""
    
    file_paths = []
    filenames = os.listdir(OUTPUT_PATH)
    for filename in filenames:
        if re.search("(^Qlik|^Export).*\.(csv|log)$", filename):
            file_paths.append(os.path.join(OUTPUT_PATH, filename))

    return file_paths


def create_folder_copy_files(file_paths):
    """create the folder and copy the files to the directory that is created"""

    global path
    
    path = '{}\\Qlik Exports Historic Data\\AIRE_Qlik_Extract_{}'.format(OUTPUT_PATH, today)
        
    if not os.path.exists(path):
        os.makedirs(path,mode=0o666)
    
         
    for file in file_paths:
        shutil.copy2(file,path)


    #for file in file_paths:
        #shutil.copy2(file,MAIN_FOLDER_PATH)


def remove_files(file_paths):
    
    #delete files
    for file in file_paths:
        os.remove(file)


def send_mail():
    pythoncom.CoInitialize()
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = "dadsdasd@.com;james@.com"
    mail.Subject = f"AIRE Weekly Export : {get_timestamp()}"
    mail.HTMLBody = "Hello there,<br><br>"
    mail.HTMLBody += "Summary of execution:<br><br>"
    mail.HTMLBody += mail_body
    mail.HTMLBody += "<br>Regards,<br>Exports"
    mail.Send()
    log_text("Mail sent")        

def main():

    log_text("===== START =====")

    remove_previous(OUTPUT_PATH)

    user_token = connector.login_token()

    run_actions(ActionsInfo,FilesInfo,user_token)

    send_mail()

    log_text("=====  END  =====")

    create_folder_copy_files(find_file_paths())
    #remove_files(find_file_paths())

    connector.logout_token(user_token)


if __name__ == "__main__":

        FilesInfo = [{
            "id" : "",
            "name" : ""
        }, {
            "id" : "",
            "name" : ""
        }, {
            "id" : "",
            "name" : ""
        }, {
            "id" : "",
            "name" : ""
        }]
     

    
        ActionsInfo = [{
            "id" : "",
            "name" : "",
            "type" : "Exports"
        }, {
            "id" : "",
            "name" : "",
            "type" : "Exports"
        }, {
            "id" : "",
            "name" : "",
            "type" : "Exports"
        }]
        
        main()   
