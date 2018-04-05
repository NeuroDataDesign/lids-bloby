import numpy as np
from skimage import io
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import ChannelResource
import cmd
import sys
import configparser

class NeuroDataResource:
    def __init__(self, host, token, collection, experiment, chanList):
        self._collection = collection
        self._experiment = experiment
        self._bossRemote = BossRemote({'protocol':'https',
                                       'host':host,
                                       'token':token})
        self._chanList = {}
        for chanDict in chanList:
            try:
                self._chanList[chanDict['name']] = ChannelResource(chanDict['name'],
                                                                   collection,
                                                                   experiment,
                                                                   'image',
                                                                   datatype=chanDict['dtype'])
            except:
                #TODO error handle here
                raise Exception("Failed to load")
                sys.exit(1)

    def assert_channel_exists(self, channel):
        return channel in self._chanList.keys()


    def get_cutout(self, chan, zRange=None, yRange=None, xRange=None, resolution=None):
        if not chan in self._chanList.keys():
            print('Error: Channel Not Found in this Resource')
            sys.exit(1)
            return
        if zRange is None or yRange is None or xRange is None:
            print('Error: You must supply zRange, yRange, xRange kwargs in list format')
            sys.exit(1)
        if resolution is None:
            resolution = 0
        data = self._bossRemote.get_cutout(self._chanList[chan],
                                           resolution,
                                           xRange,
                                           yRange,
                                           zRange)
        return data

def save_data(data, filename):
    try:
        io.imsave(filename, data)
    except:
        raise Exception("Data could not be saved")

def cast_uint8(data, dtype):
    data = data.astype(dtype)
    return data

def fetch_data_from_boss(params):
    config = configparser.ConfigParser()
    config.read(params['config'])
    
    host, token = [config['Default']['host'], config['Default']['token']]
    myResource = NeuroDataResource(host,
                                  token,
                                  params['collection'],
                                  params['experiment'],
                                  [{'name': params['channel'], 'dtype': params['dtype']}])

    data = myResource.get_cutout(params['channel'],
                               params['z_range'],
                               params['y_range'],
                               params['x_range'],
                               params['resolution'])

    data = cast_uint8(data, params['dtype'])
    save_data(data, params['filename'])
    print('Successfully saved image to {}'.format(params['filename']))
