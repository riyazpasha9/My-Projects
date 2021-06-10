import win32com.client
import pandas as pd
import sys


sys.path.append("C:\\Users\\pashar1\\Desktop\\Test folder\\New Anaplan_API Scripts")

import no_proxyauth_Anaplan_Connector as connector

OUTPUT_PATH = 'Y:\ACOE Project DOCS\Team Members Folder\Riyaz\Workday Files'
OK = "SUCCESSFUL"
NOT_OK = "FAILED"

MODEL = {"WorkspaceID":"" , "ModelID":""}


def password_remove():
    #xlCSVWindows	23	Windows CSV	*.csv
    excel = win32com.client.Dispatch('Excel.Application')
    workbook = excel.Workbooks.Open(f'{OUTPUT_PATH}\\AUK.xlsx',False, True, None, 'Ma021')
    xlCSVWindows = 0x17
    workbook.SaveAs(f'{OUTPUT_PATH}\\Future Leavers.csv', FileFormat = 23, Password = None)
    workbook.Close()
    print("The file has been saved")


def run_actions(ActionsInfo, FilesInfo, user_token):

        
            ret = connector.run_anaplan_action(MODEL["WorkspaceID"], MODEL["ModelID"], user_token, ActionsInfo[0]['id'])
            if (ret is None) or (not ret.ok):
                report("Action FAILED: {}".format(ActionsInfo[0]["name"]))


            model_file_path = '{}\\{}.csv'.format(OUTPUT_PATH,FilesInfo[0]['name'])
            download = connector.download_file(MODEL["WorkspaceID"],  MODEL["ModelID"] ,user_token , FilesInfo[0]["id"],model_file_path)
            if download is True:
                print(f"Download {OK}: {FilesInfo[0]['name']}")

            else:
                print(f"Download {NOT_OK}: {FilesInfo[0]['name']}")




def workday_process():

    df = pd.read_csv(r'Y:\ACOE Project DOCS\Team Members Folder\Riyaz\Workday Files\Future Leavers.csv',header = 1,encoding='cp1252')

    print(df.columns)
    print(df.shape)
    print("\n")


    print(df.isna().sum(axis=0))

    df1=df[df['Email - Work'].isna()==False]
    print(df1.shape)

    df_users = pd.read_csv('Y:\\ACOE Project DOCS\\Team Members Folder\\Riyaz\\Workday Files\\User_Access_Export.csv')
    print(df_users.columns)
    print(df_users.shape)

    df_users.rename(columns={'Unnamed: 0':'Email'},inplace=True)
    print(df_users.columns)
    print(df_users.info())

    #df_users['U0_Users.S - Active'] = df_users['U0_Users.S - Active'].astype(str)
    #df_users['U0_Users.S - Active'] = df_users['U0_Users.S - Active'].str.strip()
    print("\n")
    print(df_users.info())
    df_users_emailk=pd.DataFrame()
    df_users_email=df_users[df_users['U0_Users.S - Active']==True]['Email'].str.lower()
    #df_users_email['Email'].str.lower()
    print(df_users_email.head())

    df1_Emailwork=pd.DataFrame()
    df1_Emailwork['Email']=df1['Email - Work'].str.lower()
    df2_finalmails=pd.merge(df1_Emailwork,df_users_email,on='Email',how='inner')

    for email in df2_finalmails['Email']:
        if not email=='':
            df2_finalmails['Terminated?'] = True

    df2_finalmails.to_csv('Y:\ACOE Project DOCS\Team Members Folder\Riyaz\Workday Files\\Email Addresses not found in WD.csv',index=False)
    print("The Workday file has been sucessfully created")

    print(df2_finalmails.shape)
    print(df2_finalmails.isna().sum())

def main():

    print("===== START =====")

    password_remove()

    user_token = connector.login_token()

    run_actions(ActionsInfo,FilesInfo,user_token)

    connector.logout_token(user_token)

    workday_process()

    print("=====  END  =====")

    
                                                                

if __name__ == "__main__":

        FilesInfo = [{
            "id" : "",
            "name" : "User_Access_Export"
        }]
     

    
        ActionsInfo = [{
            "id" : "",
            "name" : "User_Access_Export",
            "type" : "Exports"
        }]
        
        main()                                                                
