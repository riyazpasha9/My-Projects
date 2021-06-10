import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.backends import default_backend
import hashlib
import json
import os
import requests
import sys
from urllib3.util.retry import Retry
from base64 import b64encode
import os
import requests
from OpenSSL import crypto
import random
import string


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
    return ('C:\\Users\\' + RACFID + '\\AppData\\Local\\PublicKey.pem')
	
def get_private_key_path():
    
    RACFID = os.environ.get('USERNAME')
    return ('C:\\Users\\' + RACFID + '\\AppData\\Local\\Privatekey.pem')

def get_generic_post_data():
    
    return {'localeName': 'en_US'}

def get_proxies():
      return {'http': proxy_IP, 'https': proxy_IP, 'ftp': proxy_IP}
#Certificate handling	
def read_certificate(path_to_cert):

    with open(path_to_cert, "r") as my_pem_file:
            my_pem_text = my_pem_file.read()
    return {'Authorization': 'CACertificate ' + base64.b64encode(my_pem_text.encode('utf-8')).decode('utf-8'),'Content-Type': 'application/json'}
    
    
#Loading the Private Key and Signing the Rand_Bytes with the Privatekey and encoding it.
"""
def generate_string_and_sign(path_to_priv_key):

    rand_bytes = os.urandom(100)
    rand_string = base64.b64encode(rand_bytes).decode('utf-8')
    with open(path_to_priv_key, 'rb') as priv_key_file:
        RSA_private_key = serialization.load_pem_private_key(priv_key_file.read(), password =None)
    rand_signed_string = RSA_private_key.sign(rand_bytes,padding.PSS(mgf = padding.MGF1(hashes.SHA256()), salt_length = padding.PSS.MAX_LENGTH), hashes.SHA256())
    rand_signed_string = base64.b64encode(rand_signed_string).decode('utf-8')

                        
    return {"encodedData" : rand_string, "encodedSignedData" : rand_signed_string}
"""

def generate_string_and_sign(privateKey):

    # usage
    data = os.urandom(150)
    key = load_pem_private_key(open(privateKey).read().encode('utf-8'), None, backend=default_backend())
    backend = default_backend()
    signature = key.sign(
        data,
        padding.PKCS1v15(
        )
        ,
        hashes.SHA512()
    )
    # print(signature)
    signed_nonce = base64.b64encode(signature).decode('utf-8')  # base64.b64encode(data).decode('utf-8'))
    json_string = '{ "encodedData":"' + str(
        base64.b64encode(data).decode('utf-8')) + '", "encodedSignedData":"' + signed_nonce + '"}'
    # print(json_string)
    return json_string

#Auth Anaplan API
def login_token(verbose = True):
    
    url = 'https://auth.anaplan.com/token/authenticate'

    if verbose:
        print('Requesting for auth token using Certificates')

    
    request = requests.post(url,headers=read_certificate(get_cert_path()),data = json.dumps(generate_string_and_sign(get_private_key_path())),proxies=get_proxies())

    if request.ok:
        if verbose:
            print('Authorization SUCCESSFUL, status code: {}'.format(request.status_code))
    else:
        if verbose:
            print('Authorization FAILED, status code: {}'.format(request.status_code))
        return ''
    
    return "AnaplanAuthToken " + json.loads(request.text)["tokenInfo"]["tokenValue"]

def refresh_token(verbose = True):
    

    url = 'https://auth.anaplan.com/token/refresh'

    if verbose:
        print('Requesting for auth token refresh...')

    request = requests.post(url,headers=read_certificate(get_cert_path()), data = json.dumps(generate_string_and_sign(get_private_key_path())),proxies=get_proxies())

    if request.ok:
        if verbose:
            print('Refresh SUCCESSFUL, status code: {}'.format(request.status_code))
    else:
        if verbose:
            print('Refresh FAILED, status code: {}'.format(request.status_code))
        return ''

    return "AnaplanAuthToken " + json.loads(request.text)["tokenInfo"]["tokenValue"]

def logout_token(headers,verbose = True):

    url = 'https://auth.anaplan.com/token/logout'

    if verbose:
        print('Logging out...')

    request = requests.post(url,headers={'Authorization':headers,'Content-Type': 'application/json'},proxies=get_proxies())

    if verbose:
        if request.ok:
            print('Logout SUCCESSFUL, status code: {}'.format(request.status_code))
        else:
            print('Logout FAILED, status code: {}'.format(request.status_code))

#Anaplan Integration API
def get_anaplan_json(url, headers, verbose = True):

    if verbose:
        print('Getting data from: ' + url)
        

    request = requests.get(url, headers={'Authorization':headers,'Content-Type': 'application/json'}, proxies=get_proxies())

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
        if verbose:
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
        if verbose:
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
    
    request = requests.post(url, data=json.dumps(post_data), headers=headers, proxies=get_proxies())

    if verbose:
        if request.ok:
            print('Action SUCCESSFUL, status code: {}'.format(request.status_code))
        else:
            print('Action FAILED, status code: {}'.format(request.status_code))
        
    return request

def mapping_file_to_table(mapping_file, verbose = True):
    
    if not os.path.isfile(mapping_file):
        if verbose:
            print('File {} does not exist.'.format(mapping_file))
        return None
    data = open(mapping_file, 'r').read().splitlines()
    if len(data)==0 or len(data)%2==1:
        if verbose:
            print('Invalid mapping data stream: {}'.format(data))
        return None
    mapping_table = []
    for i in range(len(data)//2):
        mapping_table.append([data[i],data[i+1]])
        
    return mapping_table

def mapping_table_to_stream(mapping_table):

    mapping_parameters = get_generic_post_data()
    mapping_parameters["mappingParameters"] = []
    for entity in mapping_table:
        mapping_parameters["mappingParameters"].append({"entityType":entity[0],"entityName":entity[1]})

    return mapping_parameters
