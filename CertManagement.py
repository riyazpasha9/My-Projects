#!/usr/bin/python

"""
Simple UI to interact with AnaplanConnector Certificate
"""
import datetime
import os

import AnaplanAPIConnector_v2_0 as connector

def check_creds():
    
    print("\nChecking credentials in Anaplan...")
    try:
        connector.auto_login_token()
    except Exception as e:
        print(e)

def main():
    
    cert_path = connector.get_cert_path();
    if not os.path.isfile(cert_path):
        print("\nCertificate does not exist!\n")
        if "y" == (input("Do you want to create your certificate? (Y/N)")).lower():
            connector.generate_certificate(cert_path)
    modified = datetime.datetime.fromtimestamp(os.path.getmtime(cert_path))
    created = datetime.datetime.fromtimestamp(os.path.getctime(cert_path))
    print("\nYour certificate info:\n")
    print("Path to file:\t{}".format(cert_path))
    print("Created:\t{}\t({} days ago)".format(created, (datetime.datetime.now() - created).days))
    print("Last modified:\t{}\t({} days ago)".format(modified, (datetime.datetime.now() - modified).days))

    check_creds()

    while "y" == (input("\nDo you want to change your certificate? (Y/N)")).lower():
        connector.generate_certificate(cert_path)
        check_creds()
        
if __name__ == "__main__":
    main()
