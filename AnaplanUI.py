#!/usr/bin/python

"""
Simple UI to interact with AnaplanConnector
"""
import argparse
import os

import AnaplanAPIConnector_v2_0 as connector

def main():
    
    ws_id = ''
    model_id = ''
    action_id = ''
    file_name = ''
    file_id = ''
    mapping_file = ''

    parser = argparse.ArgumentParser(description = "Simple UI to interact with AnaplanConnector.")
    parser.add_argument("-w", "--wsid", help="provide workspace id")
    parser.add_argument("-m", "--modelid", help="provide model id")
    parser.add_argument("-a", "--actionid", help="provide action id")
    parser.add_argument("-f", "--file", help="provide path to file to upload or download (for upload or download)")
    parser.add_argument("-n", "--mapping", help="provide mapping stream file path (for import or process with mapping)")
    parser.add_argument("-p", "--passw", help="provide certificate file path (optional)")

    args = parser.parse_args()
        
    if args.wsid:
        ws_id = args.wsid
    else:
        print("Workspace ID is required!")
        exit
    if args.modelid:
        model_id = args.modelid
    else:
        print("Model ID is required!")
        exit
    if args.actionid:
        action_id = args.actionid
    else:
        print("Action ID is required!")
        exit
        
    if args.file:
        file_name = args.file
    if args.mapping:  
        mapping_file = args.mapping
        
    user = connector.login_token(read_credentials(args.passw)) if args.passw else connector.auto_login_token()

    if file_name!='':
        connector.process_anaplan_file(ws_id, model_id, user, action_id, file_name)
    else:
        if mapping_file!='':
            connector.run_anaplan_action(ws_id, model_id, user, action_id, connector.mapping_table_to_stream(connector.mapping_file_to_table(mapping_file)))
        else:
            connector.run_anaplan_action(ws_id, model_id, user, action_id)

    connector.logout_token(connector.make_session({'Authorization': user}))
        
if __name__ == "__main__":
    main()
