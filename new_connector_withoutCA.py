#!/usr/bin/python

"""
Simple script to run actions with Anaplan API
"""
#import asyncio
import base64
import getpass
import json
import os
import requests
import sys
from urllib3.util.retry import Retry
import urllib.request

#Get atributes
def get_action_family(action_id):
    
    lookup_table = [{
        'id':'112',
        'type':'imports',
        'lambda':None
    }, {
        'id':'113',
        'type': None,
        'lambda':lambda ws_id, model_id, file_id, file_name, user, verbose: upload_file(ws_id, model_id, file_id, file_name, user, verbose)
    }, {
        'id':'116',
        'type':'exports',
        'lambda':lambda ws_id, model_id, file_id, file_name, user, verbose: download_file(ws_id, model_id, file_id, file_name, user, verbose)
    }, {
        'id':'117',
        'type':'actions',
        'lambda':None
    }, {
        'id':'118',
        'type':'processes',
        'lambda':None
    }]

    return next((i for i in lookup_table if i['id']==action_id[:3]), None)

def get_cert_path():
    
    RACFID = os.environ.get('USERNAME')
    return ('C:\\Users\\' + RACFID + '\\AppData\\Local\\AnaplanCert.crt')

def get_generic_post_data():
    
    return {'localeName': 'en_US'}

def get_proxies():
    
    return {'http': proxy_ip,
                     'https': proxy_ip,
        'ftp': proxy_ip}

#Certificate handling
def write_credentials(uname, pword, pass_file):

    cert = bytes('Basic ' + str(base64.b64encode((uname + ":" + pword).encode('utf-8')).decode('utf-8')), "utf-8")
    with open(pass_file, 'wb') as file:
        file.write(cert)

def read_credentials(pass_file):

    if os.path.isfile(pass_file):
        with open(pass_file, 'r') as file:
            return file.read()
    else:
        print(pass_file, ' does not exist.')
        generate_certificate(pass_file)
        return read_credentials(pass_file)

def generate_certificate(pass_file):

    print('This program will generate certificate to authenticate logging into Anaplan through API.')
    print('The certificate will be stored here: ' + pass_file)
    print('\n   ***   Please provide your non-SSO credentials.   ***\n')
    uname1 = ''
    uname2 = ' '
    while uname1 != uname2:
        uname1 = input("Enter your email address: ")
        uname2 = input("Confirm your email address: ")
        if uname1 != uname2:
            print('Given emails are not equal!\nTry again.')
    pword1 = ''
    pword2 = ' '
    while pword1 != pword2:
        pword1 = getpass.getpass("Enter your non-SSO password: ")
        pword2 = getpass.getpass("Confirm your non-SSO password: ")
        if pword1 != pword2:
            print('Given passwords are not equal!\nTry again.')
        
    write_credentials(uname1, pword1, pass_file)

def decrypt_credentials(creds):
    return base64.b64decode(creds[6:]).decode('utf-8').split(':')

#Auth Anaplan API

def login_token(verbose = True):
    
    url = 'https://auth.anaplan.com/token/authenticate'

    if verbose:
        print('Requesting for auth token using basic auth...')

    request = requests.post(url,headers={'Authorization': read_credentials(get_cert_path())},proxies=get_proxies())

    if verbose:
        if request.ok:
            print('Authorization SUCCESSFUL, status code: {}'.format(request.status_code))
        else:
            print('Authorization FAILED, status code: {}'.format(request.status_code))
            return ''
    
    return "AnaplanAuthToken " + json.loads(request.text)["tokenInfo"]["tokenValue"]

def refresh_token(verbose = True):

    url = 'https://auth.anaplan.com/token/refresh'

    if verbose:
        print('Requesting for auth token refresh...')

    request = requests.post(url,headers={'Authorization': read_credentials(get_cert_path())},proxies=get_proxies())

    if verbose:
        if request.ok:
            print('Refresh SUCCESSFUL, status code: {}'.format(request.status_code))
        else:
            print('Refresh FAILED, status code: {}'.format(request.status_code))
            return ''

    return "AnaplanAuthToken " + json.loads(request.text)["tokenInfo"]["tokenValue"]

def logout_token(headers,verbose = True):

    url = 'https://auth.anaplan.com/token/logout'

    if verbose:
        print('Logging out...')

    request =  requests.post(url,headers=headers,proxies=get_proxies())

    if verbose:
        if request.ok:
            print('Logout SUCCESSFUL, status code: {}'.format(request.status_code))
        else:
            print('Logout FAILED, status code: {}'.format(request.status_code))

#Anaplan Integration API
def get_anaplan_json(url, headers, verbose = True):

    if verbose:
        print('Getting data from: ' + url)

    request = requests.get(url, headers=headers, proxies=get_proxies())
        
      

    if verbose:
        if request.ok:
            print('Get SUCCESSFUL, status code: {}'.format(request.status_code))
        else:
            print('Get FAILED, status code: {}'.format(request.status_code))

    return request

def upload_file(ws_id, model_id, user, file_id, file_name, verbose = True):

    headers = {
        'Authorization':user,
        'Content-Type':'application/octet-stream'
    }
    
    url = 'https://api.anaplan.com/2/0/workspaces/{}/models/{}/files/{}'.format(ws_id,model_id,file_id)    

    if not os.path.isfile(file_name):
        print('File {} does not exist.'.format(file_name))
        return None

    if verbose:
        print( 'Uploading file...\nWorkspace ID: {}\nModel ID: {}\nFile ID: {}\nFile name: {}'.format(ws_id, model_id, file_id, file_name))
    
    dataFile = open(file_name, 'r', encoding='utf-8').read().encode('utf-8')

    request = requests.put(url, headers=headers, data=(dataFile), proxies=get_proxies())
    

    if verbose:
        if request.ok:
            print('Upload SUCCESSFUL, status code: {}'.format(request.status_code))
        else:
            print('Upload FAILED, status code: {}'.format(request.status_code))
                    
    return request

def download_file(ws_id, model_id, user, file_id, file_name, verbose = True):

    getHeaders = {
        'Authorization': user
    }

    downloadHeaders = {
        'Authorization':user,
        'Accept':'application/octet-stream'
    }

    url = 'https://api.anaplan.com/2/0/workspaces/{}/models/{}/files/{}/chunks'.format(ws_id, model_id, file_id)

    if verbose:
        print('Downloading file...\nWorkspace ID: {}\nModel ID: {}\nFile ID: {}\nFile name: {}'.format(ws_id, model_id, file_id, file_name))

    config = requests.get(url, headers=getHeaders, proxies=get_proxies())
       

    if not config.ok:
        print('Unable to reach: {} Error: {}'.format(url, config.status_code))
        return None
    
    chunks = json.loads(config.text.encode('utf-8'))
    if chunks["meta"]["paging"]["currentPageSize"] == 0:
        if verbose:
            print('Cannot download the file for model {}, 0 chunks'.format(model_id))
        return None
              
    chunks = chunks["chunks"]

    with open(file_name, 'wb') as output:
        for i in chunks:
            chunkID = i["id"]
            urlChunk = url + f'/{chunkID}'
            getChunk = requests.get(urlChunk, headers=downloadHeaders, proxies=get_proxies())
            output.write(getChunk.content)
            if verbose:
                print('Chunk {} download status code: {}'.format(chunkID, getChunk.status_code))

    return True

def process_anaplan_file(ws_id, model_id, user, file_id, file_name, verbose = True):

    action_family = get_action_family(file_id)['lambda']
    if action_family is None:
        if verbose:
            print('Incorrect File ID!')
        return None
    return action_family(ws_id, model_id, user, file_id, file_name, verbose)


def run_anaplan_action(ws_id, model_id, user, action_id, post_data = None, verbose = True):

    headers = {
        'Authorization': user,
        'Content-Type': 'application/json'
    }

    if post_data is None:
        post_data = get_generic_post_data()

    action_family = get_action_family(action_id)['type']
    if action_family is None:
        if verbose:
            print('Incorrect Action ID!')
        return None

    url = 'https://api.anaplan.com/2/0/workspaces/{}/models/{}/{}/{}/tasks'.format(ws_id, model_id, action_family, action_id)

    if verbose:
        print( 'Running action...\nWorkspace ID: {}\nModel ID: {}\nAction Family: {}\nAction ID: {}\nPost data: {}'.format(ws_id, model_id, action_family, action_id, post_data))

    request = requests.post(url, data=json.dumps(post_data), headers=headers,proxies=get_proxies())
    

    if verbose:
        if request.ok:
            print('Action SUCCESSFUL, status code: {}'.format(request.status_code))
        else:
            print('Action FAILED, status code: {}'.format(request.status_code))
        
    return request

def mapping_file_to_table(mapping_file):
    
    if not os.path.isfile(mapping_file):
        print('File {} does not exist.'.format(mapping_file))
        return None
    data = open(mapping_file, 'r').read().splitlines()
    if len(data)==0 or len(data)%2==1:
        print('Invalid mapping data stream: {}'.format(data))
        return None
    mapping_table = []
    for i in range(len(data)//2):
        mapping_table.append([data[i],data[i+1]])
        
    return mapping_table

def mapping_table_to_stream(mapping_table):

    mapping_parameters = []
    for entity in mapping_table:
        mapping_parameters.append({"entityType":entity[0],"entityName":entity[1]})

    return {'localeName':'en_US', "mappingParameters":mapping_parameters}
