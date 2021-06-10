"""

Riyaz Pasha

This script will replace the manual process for the Anaplan activity. The two files shared by the Craig Peters will be merged into one file.
The ModelID and Workspace ID are pulled into the main file Anaplan Activity and the last model activity column is formated to ISO Date format.

"""

import shutil
import glob
import re
import os
import sys
import pandas as pd
from datetime import datetime
import zipfile

sys.path.append("C:\\Users\\pashar1\\Desktop\\Test folder\\New Anaplan_API Scripts")

import new_AnaplanConnectorAPI_CA_version as connector

inputfile = 'C:\\Users\\pashar1\\Downloads'
outputpath = 'Y:\\ACOE Project DOCS\\Regular BAU\\Monthly Anaplan User activity'
CMH_MODEL = {"WorkspaceID":"" , "ModelID":""}
today = datetime.now()
month = today.strftime("%B")[0:3]
year = today.strftime("%Y")

OK = "SUCCESSFUL"
NOT_OK = "FAILED"


def find_file_paths():
    """Lists all files with matching names in ARCHIVE_PATH"""
    
    file_paths = []
    filenames = os.listdir(inputfile)
    for filename in filenames:
        if re.search("(^Anaplan Activity|[User Workspace|Reports|Anaplan]).*\.(zip)$", filename):
            file_paths.append(os.path.join(inputfile, filename))

    return file_paths



def create_folder_copy_files(file_paths):
    """create the folder and copy the files to the directory that is created"""

    global path
    
    path = "{}\\{} {}".format(outputpath, month,year)
        
    if not os.path.exists(path):
        os.makedirs(path,mode=0o666)
    
         
    for file in file_paths:
        shutil.move(file,path)

# to unzip the files
    files=os.listdir(path)
    for file in files:
        if file.endswith('.zip'):
            filePath=path+'/'+file
            zip_file = zipfile.ZipFile(filePath)
            for names in zip_file.namelist():
                zip_file.extract(names,path)
            zip_file.close()
    

def anaplan_user_activity():
    

    filenames = os.listdir(path)
    """
    print([f'{path}\\{filename}' for filename in filenames if re.search("(Workpace|Model|Model_by_Wspace|Workspace_by|Workspace_&amp).*.(csv)$", filename)])
    print("\n")
    print([f'{path}\\{filename}' for filename in filenames if re.search("User_Activity.*.(csv)$", filename)])
    """

    df1 = pd.read_csv([f'{path}\\{filename}' for filename in filenames if re.search("(User_Activity|Users).*.(csv)$", filename)][0])
    print(df1.head())
    print(f"Rows {df1.shape[0]} and Columns {df1.shape[1]}")
    print("\n")

    df2 = pd.read_csv([f'{path}\\{filename}' for filename in filenames if re.search("(Workpace|Model|Model_by_Wspace|Workspace_by|Workspace_&amp|model_size).*.(csv)$", filename)][0])
    print(df2.head())
    print(f"Rows {df2.shape[0]} and Columns {df2.shape[1]}")
    print("\n")

    df2.rename(columns={'workspaceName':'Workspace Name','modelGuid':'Model ID','workspaceGuid':'Workspace ID','modelName':'Model Name'},inplace=True)
    print(df2.columns)
    df3 = df1.merge(df2,how ='left')
    df3.drop_duplicates()
    print(df3.shape)
    print(df3.columns)

    cols = list(df3.columns.values) #Make a list of all of the columns in the df
    cols.pop(cols.index('Model ID')) #Remove ModelID from list
    cols.pop(cols.index('Workspace ID')) #Remove Workspace ID from list
    df3 = df3[cols+['Model ID','Workspace ID']]
    #print(df3.columns)

    df3['Last Model Activity'] = pd.to_datetime(df3['Last Model Activity'],errors='coerce').apply(lambda x: x.isoformat()[:16]).replace('NaT','-')
    print(df3['Last Model Activity'].head())

    df3.to_csv(f'{path}\\Aviva_user_activity_MONTH.csv',index=False)
    print("CSV file created sucessfully")


def upload_files(FILES_INFO, user_token):
    """Uploads files to the ACOE model"""

    for file in FILES_INFO:
        
            request = connector.upload_file(CMH_MODEL["WorkspaceID"],
                                            CMH_MODEL["ModelID"],
                                            user_token,
                                            file["id"],
                                            f"{path}\\{file['name']}.csv")
            
            print(f"Upload {file['name']} {OK if request.ok else NOT_OK}, status code: {request.status_code}")
            
def run_actions(ACTIONS_INFO, user_token):
    """Runs actions in Anaplan models"""
    
    for action in ACTIONS_INFO:
        
            request = connector.run_anaplan_action(ACTIONS_INFO[action]["model"]["WorkspaceID"],
                                                   ACTIONS_INFO[action]["model"]["ModelID"],
                                                   user_token,
                                                   ACTIONS_INFO[action]["id"])

            print(f"Process {action} {OK if request.ok else NOT_OK}, status code: {request.status_code}")

    
    
def main():

    user_token = connector.login_token()
    files=find_file_paths()

    create_folder_copy_files(files)
    anaplan_user_activity()

    upload_files(FILES_INFO, user_token)
    run_actions(ACTIONS_INFO, user_token)

    connector.logout_token(user_token)


if __name__ == "__main__":

    FILES_INFO = [
        {"id" : "113000000205","name" : "MONTH"}
        ]
    ACTIONS_INFO = {'Import Last Model Activity':{'model':CMH_MODEL,'id':'112000000267'}}
                    

    main()


        
