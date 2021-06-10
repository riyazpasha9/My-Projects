#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

----------------
"""
import csv
import datetime
import sys
import os

sys.path.append("C:\\Users\\pashar1\\Desktop\\Test folder\\New Anaplan_API Scripts")

import no_proxyauth_Anaplan_Connector as connector

# the below models are ARCHIVED
"""
MODELS = [
    {"Name" : "PROD", "WorkspaceID" : "" , "ModelID" : ""}
    ,{"Name" : "TEST", "WorkspaceID" : "" , "ModelID" : ""}
    ,{"Name" : "DEV", "WorkspaceID" : "" , "ModelID" : ""}
    ]
"""
'Models'
AIRE_MODELS = [
    {"Name" : "DEV", "WorkspaceID" : "" , "ModelID" : ""}
    ,{"Name" : "TEST", "WorkspaceID" : "" , "ModelID" : ""}
    ,{"Name" : "PROD", "WorkspaceID" : "" , "ModelID" : ""}
    ]


time_file_id = "113000000405"
time_file2_id = "113000000348"  #current time.csv file id
time_import_id = "112000000441"
time_import2_id = "112000000352" # Current Time Load Action

MAIN_PATH = "C:\\Users\\pashar1\\Desktop\\Test folder\\New Anaplan_API Scripts\\Real Estate - Automated Frequent Time\\"
API_URL = "https://api.anaplan.com/2/0/"

def main():

    now = datetime.datetime.now()    
    time_file = f"{MAIN_PATH}Current_Time_Frequent.csv"
    with open(time_file, 'w', encoding='utf-8', newline='') as file:
        wr = csv.writer(file, dialect='excel', quoting=csv.QUOTE_ALL)

        wr.writerow(["Line items", "Values"])
        wr.writerow(["Current date", now.strftime("%d.%m.%Y")])
        wr.writerow(["Current hour", now.strftime("%H:%M")])
        
    user = connector.login_token()

    """
    for model in MODELS:    
        connector.upload_file(model["WorkspaceID"], model["ModelID"], user, time_file_id, time_file)
        connector.run_anaplan_action(model["WorkspaceID"], model["ModelID"], user, time_import_id)
    """

    for aire in AIRE_MODELS:      
        connector.upload_file(aire["WorkspaceID"], aire["ModelID"], user, time_file2_id, time_file)
        connector.run_anaplan_action(aire["WorkspaceID"], aire["ModelID"], user, time_import2_id)

    connector.logout_token(user)

if __name__ == "__main__":
    
    main()
