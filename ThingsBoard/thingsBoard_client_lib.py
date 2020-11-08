from json import load
import logging
import uuid
# Importing models and REST client class from Community Edition version
from tb_rest_client.rest_client_ce import *
# Importing the API exception
from tb_rest_client.rest import ApiException


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# ThingsBoard REST API URL
url = "http://83.212.126.9:9090"
# Default Tenant Administrator credentials
username = "tenant@thingsboard.org"
password = "tenant"


# Creating the REST client object with context manager to get auto token refresh
with RestClientCE(base_url=url) as rest_client:
    try:
        # Auth with credentials
        rest_client.login(username=username, password=password)
        deviceName = uuid.uuid4().hex
        #create devices
        deviceIds = ['8e070a00-2148-11eb-ae92-1b95bbb7bc6f', '8e17abd0-2148-11eb-ae92-1b95bbb7bc6f', '8e282690-2148-11eb-ae92-1b95bbb7bc6f', '8e38c860-2148-11eb-ae92-1b95bbb7bc6f', '8e499140-2148-11eb-ae92-1b95bbb7bc6f']
        # for i in range(5):
        #     device = Device(name=deviceName[:6]+str(i), type="default")
        #     device = rest_client.save_device(device)
        #     deviceIds.append(device.id.id)
        # print(deviceIds)
        with open("dashboard.json", "r") as dashboard_file:
            dashboard_json = load(dashboard_file)
            dashboard_json['configuration']['entityAliases']['dd60776e-9532-7438-8311-77197aa92a87']['filter']['entityList'] = deviceIds
            print(dashboard_json)
        # I have no idea what I'm doing, AutoMapper?!
        dashboard = Dashboard(
            title=dashboard_json["title"],
            configuration=dashboard_json["configuration"],
            id = DashboardId(
                entity_type=dashboard_json['id']['entityType'],
                id=dashboard_json['id']['id']
            ),
            name = dashboard_json["name"],
            created_time = dashboard_json["createdTime"],
            tenant_id = TenantId(
                entity_type=dashboard_json['tenantId']['entityType'],
                id=dashboard_json['tenantId']['id']
            )
        )
        print(dashboard)
        dashboard = rest_client.save_dashboard(dashboard)

    except ApiException as e:
        logging.exception(e)