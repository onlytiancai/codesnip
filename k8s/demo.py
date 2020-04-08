from kubernetes import client, config

ApiToken = "eyJh"  #ApiToken
configuration = client.Configuration()
setattr(configuration, 'verify_ssl', False)
client.Configuration.set_default(configuration)
configuration.host = "https://127.0.0.1:6443"    #ApiHost
configuration.verify_ssl = False
configuration.debug = True
configuration.api_key = {"authorization": "Bearer " + ApiToken}
client.Configuration.set_default(configuration)

k8s_api_obj  = client.CoreV1Api(client.ApiClient(configuration))
ret = k8s_api_obj.list_namespaced_pod("default")
print(ret)
