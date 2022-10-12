import os
import requests
from time import sleep

# Gets pool name from pw.conf
def get_pool_name():
    with open("pw.conf") as fp:
        Lines = fp.readlines()
        for line in Lines:
            if 'sites' in line:
                return line.split('[')[1].split(']')[0]

# Get main host from pw.conf like go.parallel.works or noaa.parallel.works
def get_main_host():
    with open("pw.conf") as fp:
        Lines = fp.readlines()
        for line in Lines:
            if 'MAIN_HOST' in line:
                return line.replace('MAIN_HOST:"','').replace('"','').strip()
    # FIXME: MAIN_HOST is still not included in pw.conf in go
    return 'https://go.parallel.works/'


def get_pool_info(pool_name, url_resources, retries = 3):
    while retries >= 0:
        res = requests.get(url_resources)
        for pool in res.json():
            # FIXME: BUG sometimes pool['name'] is None when you just started the pool
            if type(pool['name']) == str:
                if pool['name'].replace('_','') == pool_name.replace('_',''):
                    return pool
        print('Retrying get_pool_info({}, {}, retries = {})'.format(pool_name, url_resources, str(retries)))
        sleep(3)
        retries += -1
    raise(Exception('Pool name not found response: ' + pool_name))

# Get the IP of the master node in the pool
def get_master_node_ip():
    pool_name = get_pool_name()
    url_resources = get_main_host() +"/api/resources?key=" + os.environ['PW_API_KEY']
    while True:
        cluster = get_pool_info(pool_name, url_resources)

        if cluster['status'] == 'on':
            if 'masterNode' in cluster['state']:
                ip = cluster['state']['masterNode']
            else:
                ip = None
                
            if ip is None:
                print('Waiting for cluster {} to get an IP'.format(pool_name), flush = True)
            else:
                return ip
        else:
            print('Waiting for cluster {} status to be on'.format(pool_name), flush = True)
            print('Cluster status: ' + cluster['status'])

        sleep(20)


def is_coaster_pool():
    coaster_pools = ['awssh', 'gce', 'azure']
    pool_name = get_pool_name()
    url_resources = get_main_host() +"/api/resources?key=" + os.environ['PW_API_KEY']
    res = requests.get(url_resources)
    pool_info = get_pool_info(pool_name, url_resources)
    if pool_info['type'] in coaster_pools:
        return True
    else:
        return False
